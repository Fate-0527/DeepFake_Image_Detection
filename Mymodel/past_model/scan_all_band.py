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

from fire_model_binary import FIRE_model 
from config import Config
from utils import set_seed
from dataset import collect_image_paths

# -------------------------------------------------------------------
# [설정] 속도 최적화를 위해 데이터 수 약간 조정
# -------------------------------------------------------------------
IMG_SIZE = 512
SUBSET_SIZE = 30   # ★ 50 -> 30으로 감소 (속도 1.6배 향상)
BATCH_SIZE = Config.BATCH_SIZE     # VRAM 넉넉하면 8로 올리세요

# 탐색 범위 (기존 유지)
R_MIN_CANDIDATES = list(range(10, 70, 10)) 
R_MAX_CANDIDATES = list(range(100, 210, 10))

SAVE_PATH = "./result_png/grid_search_heatmap_opt.png"

# -------------------------------------------------------------------
# 데이터셋
# -------------------------------------------------------------------
class SimpleDataset(Dataset):
    def __init__(self, real_paths, fake_paths, img_size=512):
        self.paths = real_paths + fake_paths
        self.labels = [0] * len(real_paths) + [1] * len(fake_paths)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx]

def find_optimal_band():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print(f">>> Loading Model on {device}...")
    model = FIRE_model(device=device)
    model.eval()

    print(f">>> Collecting Data (Subset: {SUBSET_SIZE})...")
    real_paths = collect_image_paths(Config.REAL_DIRS)
    fake_paths = collect_image_paths(Config.FAKE_DIRS)
    random.shuffle(real_paths); random.shuffle(fake_paths)
    

    ds = SimpleDataset(real_paths, fake_paths, img_size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # -------------------------------------------------
    # ★ [최적화 1] 원본 이미지 점수(Score A) 미리 계산!
    # -------------------------------------------------
    print(">>> Pre-calculating Score A (Original Image Error)...")
    cached_score_A = []     # 점수 저장용 리스트
    cached_labels = []      # 라벨 저장용 리스트
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Caching Score A"):
            imgs = imgs.to(device)
            # 원본 오차 계산 (딱 한 번만 수행)
            err_A = model.get_noise_pred_error(imgs, t_step=Config.T_STEP)
            
            # 평균 점수로 변환하여 저장 (메모리 절약)
            score_A_batch = err_A.mean(dim=(1, 2, 3)).cpu().numpy()
            
            cached_score_A.append(score_A_batch)
            cached_labels.append(labels.cpu().numpy())
            
    # 리스트를 하나의 큰 배열로 합침
    cached_score_A = np.concatenate(cached_score_A)
    cached_labels = np.concatenate(cached_labels)

    # -------------------------------------------------
    # Grid Search Loop
    # -------------------------------------------------
    results = pd.DataFrame(index=R_MIN_CANDIDATES, columns=R_MAX_CANDIDATES)
    best_score = -1
    best_pair = (0, 0)

    # 총 반복 횟수
    total_iter = len(R_MIN_CANDIDATES) * len(R_MAX_CANDIDATES)
    pbar = tqdm(total=total_iter, desc="Grid Searching")

    for r_min in R_MIN_CANDIDATES:
        for r_max in R_MAX_CANDIDATES:
            if r_min >= r_max:
                results.at[r_min, r_max] = 0
                pbar.update(1)
                continue
            
            # 1. 마스크 생성
            rows, cols = IMG_SIZE, IMG_SIZE
            mask = torch.zeros((1, 1, rows, cols), dtype=torch.float32, device=device)
            crow, ccol = rows // 2, cols // 2
            y, x = torch.meshgrid(torch.arange(rows, device=device), 
                                  torch.arange(cols, device=device), indexing='ij')
            dist_sq = (x - ccol) ** 2 + (y - crow) ** 2
            mask_area = (dist_sq >= r_min**2) & (dist_sq < r_max**2)
            mask[:, :, mask_area] = 1.0
            inv_mask = 1.0 - mask 
            
            # 이번 조합의 점수들을 모을 리스트
            current_diffs = []

            # 2. 모델 추론 (Score B만 계산)
            with torch.no_grad():
                for imgs, _ in loader: # 라벨은 이미 캐싱됨
                    imgs = imgs.to(device)
                    
                    # PSE 생성
                    freq_image = torch.fft.fftn(imgs, dim=(-2, -1))
                    freq_image = torch.fft.fftshift(freq_image, dim=(-2, -1))
                    filtered_freq = freq_image * inv_mask
                    filtered_freq = torch.fft.ifftshift(filtered_freq, dim=(-2, -1))
                    x_pse = torch.abs(torch.fft.ifftn(filtered_freq, dim=(-2, -1))).clamp(0, 1)
                    
                    # Error B 계산
                    err_B = model.get_noise_pred_error(x_pse, t_step=Config.T_STEP)
                    score_B_batch = err_B.mean(dim=(1, 2, 3)).cpu().numpy()
                    
                    current_diffs.append(score_B_batch)
            
            # 배치를 하나로 합침
            all_score_B = np.concatenate(current_diffs)
            
            # ★ 핵심: 미리 계산해둔 A에서 방금 계산한 B를 뺌
            # (순서가 보장되므로 인덱스로 매칭 가능)
            diffs = cached_score_A - all_score_B
            
            # Real/Fake 분리
            real_scores = diffs[cached_labels == 0]
            fake_scores = diffs[cached_labels == 1]

            # 분리도 계산
            if len(real_scores) > 0 and len(fake_scores) > 0:
                score = wasserstein_distance(real_scores, fake_scores)
            else:
                score = 0
            
            results.at[r_min, r_max] = score
            
            if score > best_score:
                best_score = score
                best_pair = (r_min, r_max)
                
            pbar.set_postfix({'Best': f"{best_pair}:{best_score:.4f}", 'Curr': f"{r_min}-{r_max}"})
            pbar.update(1)
            
    pbar.close()

    # -------------------------------------------------
    # 시각화
    # -------------------------------------------------
    print("\n>>> Creating Heatmap...")
    plt.figure(figsize=(12, 10))
    results = results.astype(float)
    sns.heatmap(results, annot=True, fmt=".4f", cmap="viridis", 
                xticklabels=R_MAX_CANDIDATES, yticklabels=R_MIN_CANDIDATES)
    plt.title(f"Optimal Frequency Band Search (Max Score: {best_score:.4f})")
    plt.xlabel("R_MAX")
    plt.ylabel("R_MIN")
    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    
    print("\n" + "="*50)
    print(f"🏆 Best Pair: R_MIN={best_pair[0]}, R_MAX={best_pair[1]} (Score: {best_score:.4f})")
    print("="*50)

if __name__ == "__main__":
    find_optimal_band()