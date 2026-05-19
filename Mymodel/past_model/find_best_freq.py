import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from PIL import Image
from torchvision import transforms

# 사용자 파일 import
from fire_model import FIRE_model, fft_filter  # ★ 변경
from config import Config
from utils import set_seed
from dataset import collect_image_paths
from torch.utils.data import Dataset, DataLoader

# ★ 정규화 없는 단순 Dataset
class SimpleDataset(Dataset):
    def __init__(self, real_paths, fake_paths, img_size=512):
        self.paths = real_paths + fake_paths
        self.labels = [0] * len(real_paths) + [1] * len(fake_paths)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # 0~1 범위, 정규화 없음!
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx]

def find_best_frequency():
    # -----------------------------------------------------------
    # 1. 실험할 주파수 대역 후보군 (r_min, r_max)
    # ★ 512 해상도 기준 탐색
    # -----------------------------------------------------------
    candidates = [
        (20, 200),   # 현재 최고 (기준점)
        (20, 190),   # r_max 확장
        (10, 200)   # r_max 더 확장
    ]

    BEST_T = 100        # 최적의 t값
    # -----------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    print("Loading Model...")
    model = FIRE_model(device=device)  # ★ 변경
    model.eval()

    print("Collecting Data...")
    real_paths = collect_image_paths(Config.REAL_DIRS)
    fake_paths = collect_image_paths(Config.FAKE_DIRS)

    # 셔플 후 자르기 (데이터 편향 방지)
    random.shuffle(real_paths)
    random.shuffle(fake_paths)

    # ★ 변경: 정규화 없는 SimpleDataset 사용
    ds = SimpleDataset(real_paths, fake_paths, img_size=Config.IMG_SIZE)
    loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    print(f"\n[Experiment Start] t={BEST_T}, Candidates={candidates}")
    
    results = {}
    plt.figure(figsize=(5 * len(candidates), 5))

    for idx, (r_min, r_max) in enumerate(candidates):
        print(f"\nTesting Band: {r_min} ~ {r_max} ...")
        
        # ★ 수정: 고정 마스크 직접 생성 (ESPCN 없이)
        rows, cols = Config.IMG_SIZE, Config.IMG_SIZE
        mask = torch.zeros((1, 1, rows, cols), dtype=torch.float32, device=device)
        crow, ccol = rows // 2, cols // 2
        y, x = torch.meshgrid(torch.arange(rows, device=device), 
                              torch.arange(cols, device=device), indexing='ij')
        dist_sq = (x - ccol) ** 2 + (y - crow) ** 2
        mask_area = (dist_sq >= r_min**2) & (dist_sq < r_max**2)
        mask[:, :, mask_area] = 1.0
        inv_mask = 1.0 - mask  # 중간 주파수 제거용
        
        real_scores, fake_scores = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(loader, leave=False):
                imgs = imgs.to(device)
                
                # ★ 수정: 고정 마스크로 직접 FFT 필터링
                # 1. FFT 변환 (0~1 범위 이미지)
                freq_image = torch.fft.fftn(imgs, dim=(-2, -1))
                freq_image = torch.fft.fftshift(freq_image, dim=(-2, -1))
                
                # 2. 중간 주파수 제거 (inv_mask 적용)
                filtered_freq = freq_image * inv_mask
                
                # 3. IFFT로 복원 + clamp (0~1 범위 유지)
                filtered_freq = torch.fft.ifftshift(filtered_freq, dim=(-2, -1))
                x_pse = torch.abs(torch.fft.ifftn(filtered_freq, dim=(-2, -1)))
                
                # ★ clamp: 0~1 범위 밖 값만 잘라냄 (원본 값 보존)
                x_pse = x_pse.clamp(0, 1)
                
                # 4. 노이즈 예측 오차 계산 (둘 다 0~1 범위)
                err_A = model.get_noise_pred_error(imgs, t_step=BEST_T)      # 원본
                err_B = model.get_noise_pred_error(x_pse, t_step=BEST_T)     # PSE
                
                # 5. 점수 계산 (평균 오차 차이)
                score_A = err_A.mean(dim=(1, 2, 3)).cpu().numpy()
                score_B = err_B.mean(dim=(1, 2, 3)).cpu().numpy()
                diff = score_A - score_B
                
                # 6. Real/Fake 나누기
                current_labels = labels.cpu().numpy()
                real_scores.extend(diff[current_labels == 0])
                fake_scores.extend(diff[current_labels == 1])

        # 점수 계산
        score = wasserstein_distance(real_scores, fake_scores)
        results[f"{r_min}-{r_max}"] = score

        # 그래프
        plt.subplot(1, len(candidates), idx+1)
        sns.kdeplot(real_scores, fill=True, color='blue', label='Real', alpha=0.3)
        sns.kdeplot(fake_scores, fill=True, color='red', label='Fake', alpha=0.3)
        plt.title(f"Band: {r_min}-{r_max}\nScore: {score:.4f}")
        plt.xlabel("Score (B - A)")
        if idx == 0: plt.legend()

    plt.tight_layout()
    plt.savefig("./result_png/freq_experi7.png")
    
    best_band = max(results, key=results.get)
    print("\n" + "="*40)
    print(f"🏆 최적의 주파수 대역: {best_band}")
    print(f"   (Score: {results[best_band]:.4f})")
    print("="*40)
    print("-> fire_model.py 또는 upgrade_train.py의 radiuslow/radiushigh를 이 값으로 바꾸세요!")

if __name__ == "__main__":
    find_best_frequency()