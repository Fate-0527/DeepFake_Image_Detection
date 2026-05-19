import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from scipy.stats import wasserstein_distance  # 분포 간 거리를 재는 함수
from PIL import Image
from torchvision import transforms
import glob
import os

# 사용자 모듈 import
from fire_model_binary import FIRE_model
from config import Config
from torch.utils.data import Dataset, DataLoader
from utils import set_seed

# ---------------------------------------------------------
# ★ [핵심] Config의 (Start, End) 튜플을 처리하는 함수 추가
# ---------------------------------------------------------
def collect_paths_from_tuples(dir_config):
    """
    Config에 정의된 {경로: (시작, 끝)} 딕셔너리를 받아 
    정확히 해당 구간의 파일 경로 리스트를 반환합니다.
    """
    all_paths = []
    
    for dir_path, slice_range in dir_config.items():
        # 1. 해당 폴더의 모든 이미지 검색
        files = glob.glob(os.path.join(dir_path, "*.*"))
        # ★ 중요: 순서를 보장하기 위해 반드시 정렬 (파일명 기준)
        files.sort()
        
        # 이미지 확장자 필터링
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        
        if not files:
            print(f"⚠️ Warning: No images found in {dir_path}")
            continue

        # 2. Config의 슬라이싱 규칙 적용
        start, end = slice_range
        
        # Python 리스트 슬라이싱 (None이면 끝까지)
        selected_files = files[start:end]
        
        all_paths.extend(selected_files)
        
    return all_paths

# ---------------------------------------------------------
# 정규화 없는 단순 Dataset (실험용)
# ---------------------------------------------------------
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
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            img = self.transform(img)
            return img, self.labels[idx]
        except Exception as e:
            print(f"Error loading {self.paths[idx]}: {e}")
            # 에러 시 까만 화면 반환 (실험 중단 방지)
            return torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE), self.labels[idx]

def run_experiment():
    # -----------------------------------------------------------
    # 1. 실험 설정
    # -----------------------------------------------------------
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 비교할 t 값 후보군
    t_candidates = [100, 120, 140, 160, 180, 200]
    
    # Config에서 값 가져오기
    IMG_SIZE = Config.IMG_SIZE     # 512
    RADIUS_LOW = Config.R_MIN      # 20
    RADIUS_HIGH = Config.R_MAX     # 200
    
    print(f">>> Device: {device}")
    print(f">>> Target Frequency Band: {RADIUS_LOW} ~ {RADIUS_HIGH}")
    print(f">>> Comparing Timesteps: {t_candidates}")

    # -----------------------------------------------------------
    # 2. 데이터 로드 (Config 튜플 반영)
    # -----------------------------------------------------------
    print(">>> 데이터 경로 수집 중...")
    
    # ★ 수정된 함수 사용
    real_paths = collect_paths_from_tuples(Config.REAL_DIRS)
    fake_paths = collect_paths_from_tuples(Config.FAKE_DIRS)
    
    print(f"📊 Experiment Data: Real {len(real_paths)}장 vs Fake {len(fake_paths)}장")

    if len(real_paths) == 0 or len(fake_paths) == 0:
        print("❌ Error: 데이터가 없습니다. 경로를 확인하세요.")
        return

    # 데이터셋 & 로더 생성
    ds = SimpleDataset(real_paths, fake_paths, img_size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    # -----------------------------------------------------------
    # 3. 모델 준비 & 마스크 생성
    # -----------------------------------------------------------
    model = FIRE_model(device=device)
    model.eval() # 학습 모드 아님

    # 고정 마스크 생성 (Config 값 기반)
    rows, cols = IMG_SIZE, IMG_SIZE
    mask = torch.zeros((1, 1, rows, cols), dtype=torch.float32, device=device)
    crow, ccol = rows // 2, cols // 2
    y, x = torch.meshgrid(torch.arange(rows, device=device), 
                          torch.arange(cols, device=device), indexing='ij')
    dist_sq = (x - ccol) ** 2 + (y - crow) ** 2
    
    # Config의 R_MIN, R_MAX 적용
    mask_area = (dist_sq >= RADIUS_LOW**2) & (dist_sq < RADIUS_HIGH**2)
    mask[:, :, mask_area] = 1.0
    inv_mask = 1.0 - mask  # 1에서 뺌 -> 중간 주파수 제거용 필터

    # -----------------------------------------------------------
    # 4. 실험 루프 (t값 별 성능 측정)
    # -----------------------------------------------------------
    results = {}
    
    # 그래프 크기 설정
    plt.figure(figsize=(5 * len(t_candidates), 5))

    for idx, t_val in enumerate(t_candidates):
        print(f"\n[Testing t={t_val}] Calculating scores...")
        real_scores = []
        fake_scores = []

        with torch.no_grad():
            for imgs, labels in tqdm(loader, leave=False):
                imgs = imgs.to(device)
                
                # 1. FFT 필터링 (중간 주파수 제거)
                fft_imgs = torch.fft.fft2(imgs, dim=(-2, -1))
                fft_shifted = torch.fft.fftshift(fft_imgs, dim=(-2, -1))
                
                # 마스크 적용
                filtered_fft = fft_shifted * inv_mask
                
                # IFFT 역변환 -> x_pse 생성
                fft_unshifted = torch.fft.ifftshift(filtered_fft, dim=(-2, -1))
                x_pse = torch.fft.ifft2(fft_unshifted, dim=(-2, -1)).real
                
                # Min-Max Normalization (0~1로 맞춤) -> 모델 입력용
                B, C, H, W = x_pse.shape
                x_pse_flat = x_pse.view(B, C, -1)
                x_min = x_pse_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
                x_max = x_pse_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                x_pse = (x_pse - x_min.squeeze(-1).unsqueeze(-1)) / (x_max.squeeze(-1).unsqueeze(-1) - x_min.squeeze(-1).unsqueeze(-1) + 1e-8)
                
                # 2. 오차 계산 (DiffLaRE 방식)
                # t_step 적용
                err_A = model.get_noise_pred_error(imgs, t_step=t_val)  # 원본 오차
                err_B = model.get_noise_pred_error(x_pse, t_step=t_val) # 필터링된 오차
                
                # 3. 점수 = (원본 오차) - (필터링된 오차)
                score_A = err_A.mean(dim=(1, 2, 3)).cpu().numpy()
                score_B = err_B.mean(dim=(1, 2, 3)).cpu().numpy()
                diff = score_A - score_B
                
                # 라벨별 저장
                current_labels = labels.cpu().numpy()
                real_scores.extend(diff[current_labels == 0])
                fake_scores.extend(diff[current_labels == 1])

        # 4. 분리도(Separability) 계산
        if len(real_scores) > 0 and len(fake_scores) > 0:
            sep_score = wasserstein_distance(real_scores, fake_scores)
            results[t_val] = sep_score
            
            # 그래프 그리기
            plt.subplot(1, len(t_candidates), idx+1)
            sns.kdeplot(real_scores, fill=True, color='blue', label='Real', alpha=0.3)
            sns.kdeplot(fake_scores, fill=True, color='red', label='Fake', alpha=0.3)
            plt.title(f"t={t_val}\nSep Score: {sep_score:.4f}")
            plt.xlabel("Score (B - A)")
            if idx == 0: plt.legend()
        else:
            print(f"⚠️ Warning: t={t_val}에서 점수 계산 실패 (데이터 부족?)")

    # 결과 저장
    plt.tight_layout()
    os.makedirs(Config.PNG_DIR, exist_ok=True)
    save_path = os.path.join(Config.PNG_DIR, "t_step_experiment.png")
    plt.savefig(save_path)
    print(f"\n[Done] 결과 그래프 저장됨: {save_path}")
    
    # 5. 최종 추천
    if results:
        best_t = max(results, key=results.get)
        print("\n" + "="*40)
        print(f"🏆 Best t_step: {best_t}")
        print(f"   (Score: {results[best_t]:.4f})")
        print("="*40)
        print(f"👉 Config.py의 T_STEP을 {best_t}로 수정하세요!")
    else:
        print("❌ 실험 결과가 없습니다.")

if __name__ == "__main__":
    run_experiment()