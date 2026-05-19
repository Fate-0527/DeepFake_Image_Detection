import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 사용자 모델 파일 import
try:
    from fire_model_binary import FIRE_model 
except ImportError:
    try:
        from fire_model import FIRE_model
    except ImportError:
        print("Model file not found. Please verify filename.")

from config import Config
from utils import set_seed
from dataset import collect_image_paths

# -------------------------------------------------------------------
# 1. 설정 (Configuration)
# -------------------------------------------------------------------
IMG_SIZE = 512
BATCH_SIZE = Config.BATCH_SIZE
FIXED_BAND = (20, 200)  # 검증할 주파수 대역
BEST_T = 100
SAVE_PATH = "./result_png/all_norm_comparison.png"

# -------------------------------------------------------------------
# 2. 확장된 정규화 함수 (8가지 기법)
# -------------------------------------------------------------------
def apply_normalization(x, method="Clamp (0,1)"):
    """
    x: [B, 3, H, W] 형태의 PSE 이미지 텐서 (양수 가정)
    """
    B, C, H, W = x.shape
    epsilon = 1e-8
    
    # 1. Clamp (Baseline)
    if method == "Clamp (0,1)":
        return x.clamp(0, 1)
        
    # 2. MinMax Scaler
    elif method == "MinMax Scaler":
        min_val = x.amin(dim=(1, 2, 3), keepdim=True)
        max_val = x.amax(dim=(1, 2, 3), keepdim=True)
        return (x - min_val) / (max_val - min_val + epsilon)
    
    # 3. Robust Scaler (Quantile)
    elif method == "Robust Scaler":
        flat_x = x.view(B, -1)
        q_low = torch.quantile(flat_x, 0.02, dim=1).view(B, 1, 1, 1)
        q_high = torch.quantile(flat_x, 0.98, dim=1).view(B, 1, 1, 1)
        x_scaled = (x - q_low) / (q_high - q_low + epsilon)
        return x_scaled.clamp(0, 1) # 이상치는 clamp

    # 4. MaxAbs Scaler
    elif method == "MaxAbs Scaler":
        max_abs = x.amax(dim=(1, 2, 3), keepdim=True)
        return x / (max_abs + epsilon)

    # 5. Standard + Sigmoid (Z-score normalization mapped to 0-1)
    elif method == "Standard + Sigmoid":
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        z_score = (x - mean) / (std + epsilon)
        return torch.sigmoid(z_score) # 0~1로 매핑

    # 6. Log1p + MinMax (Log transform first)
    elif method == "Log1p + MinMax":
        x_log = torch.log1p(x) # log(1+x)
        min_val = x_log.amin(dim=(1, 2, 3), keepdim=True)
        max_val = x_log.amax(dim=(1, 2, 3), keepdim=True)
        return (x_log - min_val) / (max_val - min_val + epsilon)

    # 7. Tanh Estimator (Soft clipping)
    elif method == "Tanh Estimator":
        return torch.tanh(x) # 0 근처는 선형, 큰 값은 1로 수렴

    # 8. L2 Normalization (Unit Norm)
    elif method == "L2 Normalize":
        # 이미지 전체를 하나의 벡터로 보고 L2 Norm으로 나눔
        flat_x = x.view(B, -1)
        norm = torch.norm(flat_x, p=2, dim=1).view(B, 1, 1, 1)
        # 값이 너무 작아지는 것을 방지하기 위해 스케일 조정 (선택적)
        # 여기서는 순수 L2 Norm만 적용
        return x / (norm + epsilon)

    else:
        return x

# -------------------------------------------------------------------
# 3. 데이터셋 및 메인 함수
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

def compare_all_normalization():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print(f">>> Loading Model on {device}...")
    model = FIRE_model(device=device)
    model.eval()

    print(">>> Collecting Data...")
    real_paths = collect_image_paths(Config.REAL_DIRS)
    fake_paths = collect_image_paths(Config.FAKE_DIRS)
    random.shuffle(real_paths); random.shuffle(fake_paths)
    
    ds = SimpleDataset(real_paths, fake_paths, img_size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 비교할 8가지 기법
    norm_methods = [
        "Clamp (0,1)", 
        "MinMax Scaler", 
        "Robust Scaler", 
        "MaxAbs Scaler",
        "Standard + Sigmoid",
        "Log1p + MinMax",
        "Tanh Estimator",
        "L2 Normalize"
    ]
    
    # 마스크 생성 (Fixed Band)
    r_min, r_max = FIXED_BAND
    rows, cols = IMG_SIZE, IMG_SIZE
    mask = torch.zeros((1, 1, rows, cols), dtype=torch.float32, device=device)
    crow, ccol = rows // 2, cols // 2
    y, x = torch.meshgrid(torch.arange(rows, device=device), 
                          torch.arange(cols, device=device), indexing='ij')
    dist_sq = (x - ccol) ** 2 + (y - crow) ** 2
    mask_area = (dist_sq >= r_min**2) & (dist_sq < r_max**2)
    mask[:, :, mask_area] = 1.0
    inv_mask = 1.0 - mask 

    results = {}
    
    # 그래프 설정: 2행 4열
    plt.figure(figsize=(20, 10))
    print(f"\n[Experiment Start] Band={FIXED_BAND}, Testing {len(norm_methods)} methods...")

    for idx, method in enumerate(norm_methods):
        print(f"[{idx+1}/{len(norm_methods)}] Testing: {method} ...")
        
        real_scores = []
        fake_scores = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(loader, leave=False):
                imgs = imgs.to(device)
                
                # FFT Processing
                freq_image = torch.fft.fftn(imgs, dim=(-2, -1))
                freq_image = torch.fft.fftshift(freq_image, dim=(-2, -1))
                filtered_freq = freq_image * inv_mask
                filtered_freq = torch.fft.ifftshift(filtered_freq, dim=(-2, -1))
                x_pse_raw = torch.abs(torch.fft.ifftn(filtered_freq, dim=(-2, -1)))
                
                # ★ 정규화 적용
                x_pse = apply_normalization(x_pse_raw, method=method)
                
                # Error Calculation
                err_A = model.get_noise_pred_error(imgs, t_step=BEST_T)
                err_B = model.get_noise_pred_error(x_pse, t_step=BEST_T)
                
                # Score Calculation
                score_A = err_A.mean(dim=(1, 2, 3)).cpu().numpy()
                score_B = err_B.mean(dim=(1, 2, 3)).cpu().numpy()
                diff = score_A - score_B
                
                lbl = labels.cpu().numpy()
                real_scores.extend(diff[lbl == 0])
                fake_scores.extend(diff[lbl == 1])
        
        # 분리도 점수
        score = wasserstein_distance(real_scores, fake_scores)
        results[method] = score
        
        # Subplot 그리기
        plt.subplot(2, 4, idx+1)
        sns.kdeplot(real_scores, fill=True, color='blue', label='Real', alpha=0.3)
        sns.kdeplot(fake_scores, fill=True, color='red', label='Fake', alpha=0.3)
        plt.title(f"{method}\nDist: {score:.4f}", fontsize=12, fontweight='bold')
        plt.xlabel("Score (Diff)")
        if idx == 0: plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"\n>>> 📊 Result Graph Saved: {SAVE_PATH}")
    
    # 최종 순위 출력
    print("\n" + "="*60)
    print("🏆 정규화 기법 분리도 순위 (높을수록 좋음)")
    print("="*60)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (m, s) in enumerate(sorted_results):
        rank_icon = "🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else f"{i+1}."
        print(f"{rank_icon} {m:<25} : {s:.5f}")
    print("="*60)
    print(f"✅ 최적의 정규화 기법: {sorted_results[0][0]}")
    print("   -> fire_model.py에 이 로직을 적용하세요!")

if __name__ == "__main__":
    compare_all_normalization()