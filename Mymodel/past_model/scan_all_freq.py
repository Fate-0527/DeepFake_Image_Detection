import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 사용자 모델 파일 import
from fire_model_binary import FIRE_model 
from config import Config
from utils import set_seed
from dataset import collect_image_paths

# -------------------------------------------------------------------
# 1. 설정 (Configuration)
# -------------------------------------------------------------------
IMG_SIZE = 512            # ★ 512 해상도
STEP_SIZE = 10            # 10 단위로 쪼개기
MAX_RADIUS = 370          # 512의 최대 반경(약 362) 커버
# SUBSET_SIZE = 50        # ★ 주석 처리 (전체 데이터 사용)
BATCH_SIZE = Config.BATCH_SIZE
SAVE_PATH = "./result_png/freq_scan_result_opt.png"

# -------------------------------------------------------------------
# 2. 데이터셋 (SimpleDataset)
# -------------------------------------------------------------------
class SimpleDataset(Dataset):
    def __init__(self, real_paths, fake_paths, img_size=512):
        self.paths = real_paths + fake_paths
        self.labels = [0] * len(real_paths) + [1] * len(fake_paths)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), # 0~1 range
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx]

# -------------------------------------------------------------------
# 3. 메인 함수
# -------------------------------------------------------------------
def scan_all_frequencies():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print(f">>> Loading Model on {device}...")
    model = FIRE_model(device=device)
    model.eval()

    print(f">>> Collecting Data...")
    real_paths = collect_image_paths(Config.REAL_DIRS)
    fake_paths = collect_image_paths(Config.FAKE_DIRS)

    # 셔플 후 샘플링 (전체 사용 시 주석 처리하거나, 필요시 개수 조절)
    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    
    # 균형 맞추기 (선택 사항)
    min_len = min(len(real_paths), len(fake_paths))
    real_paths = real_paths[:min_len]
    fake_paths = fake_paths[:min_len]
    
    print(f">>> Final Dataset: {len(real_paths)} Real, {len(fake_paths)} Fake")

    ds = SimpleDataset(real_paths, fake_paths, img_size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 전수조사 범위 생성 (0~10, 10~20, ..., 360~370)
    bands = []
    for r in range(0, MAX_RADIUS, STEP_SIZE):
        bands.append((r, r + STEP_SIZE))
    
    print(f">>> Scanning {len(bands)} frequency bands (0 to {MAX_RADIUS})...")
    
    # -------------------------------------------------
    # ★ [최적화 1] 원본 이미지 점수(Score A) 미리 계산!
    # -------------------------------------------------
    print(">>> Pre-calculating Score A (Original Image Error)...")
    cached_score_A = []     # 점수 저장용
    cached_labels = []      # 라벨 저장용
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Caching Score A"):
            imgs = imgs.to(device)
            # 원본 오차 계산 (딱 한 번만 수행)
            err_A = model.get_noise_pred_error(imgs, t_step=Config.T_STEP)
            
            # 평균 점수로 변환하여 저장
            score_A_batch = err_A.mean(dim=(1, 2, 3)).cpu().numpy()
            
            cached_score_A.append(score_A_batch)
            cached_labels.append(labels.cpu().numpy())
            
    # 하나로 합침
    cached_score_A = np.concatenate(cached_score_A)
    cached_labels = np.concatenate(cached_labels)

    # -------------------------------------------------
    # ★ Loop: 모든 주파수 대역 스캔
    # -------------------------------------------------
    plot_data = []
    band_scores = {}

    for r_min, r_max in tqdm(bands, desc="Frequency Sweeping"):
        band_name = f"{r_min}-{r_max}"
        
        # 1. 고정 마스크 생성
        rows, cols = IMG_SIZE, IMG_SIZE
        mask = torch.zeros((1, 1, rows, cols), dtype=torch.float32, device=device)
        crow, ccol = rows // 2, cols // 2
        y, x = torch.meshgrid(torch.arange(rows, device=device), 
                              torch.arange(cols, device=device), indexing='ij')
        dist_sq = (x - ccol) ** 2 + (y - crow) ** 2
        
        mask_area = (dist_sq >= r_min**2) & (dist_sq < r_max**2)
        mask[:, :, mask_area] = 1.0
        inv_mask = 1.0 - mask 
        
        current_score_B = []

        with torch.no_grad():
            for imgs, _ in loader: # 라벨은 이미 캐싱됨
                imgs = imgs.to(device)
                
                # FFT -> Filter -> IFFT
                freq_image = torch.fft.fftn(imgs, dim=(-2, -1))
                freq_image = torch.fft.fftshift(freq_image, dim=(-2, -1))
                
                filtered_freq = freq_image * inv_mask
                
                filtered_freq = torch.fft.ifftshift(filtered_freq, dim=(-2, -1))
                x_pse = torch.abs(torch.fft.ifftn(filtered_freq, dim=(-2, -1)))
                x_pse = x_pse.clamp(0, 1)
                
                # Score B만 계산
                err_B = model.get_noise_pred_error(x_pse, t_step=Config.T_STEP)
                score_B_batch = err_B.mean(dim=(1, 2, 3)).cpu().numpy()
                
                current_score_B.append(score_B_batch)

        # 배치를 하나로 합침
        all_score_B = np.concatenate(current_score_B)
        
        # ★ 차이 계산 (미리 계산한 A - 지금 계산한 B)
        diff = cached_score_A - all_score_B
        
        # 결과 수집 (Real/Fake 분리)
        real_vals = diff[cached_labels == 0]
        fake_vals = diff[cached_labels == 1]
        
        for v in real_vals:
            plot_data.append({'Band': band_name, 'Type': 'Real', 'Score': v})
        for v in fake_vals:
            plot_data.append({'Band': band_name, 'Type': 'Fake', 'Score': v})

        # 분리도 계산
        if len(real_vals) > 0 and len(fake_vals) > 0:
            dist = wasserstein_distance(real_vals, fake_vals)
            band_scores[band_name] = dist

    # -------------------------------------------------
    # 4. 시각화 (Box Plot)
    # -------------------------------------------------
    print(">>> Creating Visualization...")
    df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(24, 8))
    sns.boxplot(x='Band', y='Score', hue='Type', data=df, 
                palette={'Real': 'blue', 'Fake': 'red'}, showfliers=False)
    
    plt.xticks(rotation=45)
    plt.title(f"Frequency Band Analysis (Resolution {IMG_SIZE}x{IMG_SIZE})")
    plt.xlabel("Frequency Band (Radius)")
    plt.ylabel("Reconstruction Error Difference (Score)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f">>> 📊 Graph Saved: {SAVE_PATH}")

    # -------------------------------------------------
    # 5. Top-10 추천
    # -------------------------------------------------
    sorted_bands = sorted(band_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*50)
    print("🏆 Best Frequency Bands (Highest Separation)")
    print("="*50)
    for i, (band, score) in enumerate(sorted_bands[:10]):
        print(f"{i+1}. Band [{band}] : Score {score:.4f}")
    print("="*50)

if __name__ == "__main__":
    scan_all_frequencies()