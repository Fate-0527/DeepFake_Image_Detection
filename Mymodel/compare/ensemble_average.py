import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import glob

# 사용자 모듈 임포트
from fire_model_binary import FIRE_model
from config import Config

# --- 설정 ---
# 🌟 평균을 내어 콘텐츠를 지워야 하므로, 샘플 수가 많을수록 좋습니다. (가능하면 100장 이상)
NUM_SAMPLES = 100  
T_STEP = Config.T_STEP
SAVE_DIR = "phase5_ensemble_average"

TARGET_LOW = ("Low(30-140)", 30, 140)
TARGET_HIGH = ("High(170-220)", 170, 220)

def make_band_mask(rows, cols, r_low, r_high, device):
    crow, ccol = rows // 2, cols // 2
    y, x = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')
    dist_sq = (x - ccol) ** 2 + (y - crow) ** 2
    mask = ((dist_sq >= r_low**2) & (dist_sq < r_high**2)).float()
    return mask.unsqueeze(0).unsqueeze(0).to(device)

def remove_band(image, r_low, r_high, device):
    B, C, H, W = image.shape
    freq = torch.fft.fftn(image, dim=(-2, -1))
    freq = torch.fft.fftshift(freq, dim=(-2, -1))
    band_mask = make_band_mask(H, W, r_low, r_high, device)
    freq_filtered = freq * (1.0 - band_mask)
    img_filtered = torch.abs(torch.fft.ifftn(
        torch.fft.ifftshift(freq_filtered, dim=(-2, -1)), dim=(-2, -1)
    ))
    flat = img_filtered.view(B, -1)
    norm = torch.norm(flat, p=2, dim=1).view(B, 1, 1, 1)
    return img_filtered / (norm + 1e-8)

def run_phase5_ensemble_average(real_dir, fake_dir, checkpoint_path, num_samples=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FIRE_model(device=device)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        print(f"✅ Checkpoint 로드 완료")
    model.eval()

    os.makedirs(SAVE_DIR, exist_ok=True)
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    real_files = sorted([f for f in glob.glob(os.path.join(real_dir, "*.*")) if f.lower().endswith(valid_exts)])[:num_samples]
    fake_files = sorted([f for f in glob.glob(os.path.join(fake_dir, "*.*")) if f.lower().endswith(valid_exts)])[:num_samples]
    num_pairs = min(len(real_files), len(fake_files))

    print(f"📸 Phase 5: 앙상블 평균 오차 맵(지문) 추출 시작 (각 {num_pairs}장 누적)")

    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # 누적할 빈 배열 생성 (64x64 사이즈를 가정)
    sum_real_diff = None
    sum_fake_diff = None
    real_count = 0
    fake_count = 0

    with torch.no_grad():
        # --- Real 데이터 누적 ---
        for i, fpath in enumerate(real_files[:num_pairs]):
            try:
                img = Image.open(fpath).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                _, l_min, l_max = TARGET_LOW
                x_low_removed = remove_band(img_tensor, l_min, l_max, device)
                err_low_map = model.get_noise_pred_error(x_low_removed, t_step=T_STEP).squeeze(0).mean(dim=0).cpu().numpy()

                _, h_min, h_max = TARGET_HIGH
                x_high_removed = remove_band(img_tensor, h_min, h_max, device)
                err_high_map = model.get_noise_pred_error(x_high_removed, t_step=T_STEP).squeeze(0).mean(dim=0).cpu().numpy()

                diff_map = err_low_map - err_high_map

                if sum_real_diff is None:
                    sum_real_diff = np.zeros_like(diff_map)
                sum_real_diff += diff_map
                real_count += 1
            except Exception as e:
                continue
            
            if (i+1) % 20 == 0: print(f"  - Real 누적 진행률: {i+1}/{num_pairs}")

        # --- Fake 데이터 누적 ---
        for i, fpath in enumerate(fake_files[:num_pairs]):
            try:
                img = Image.open(fpath).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                _, l_min, l_max = TARGET_LOW
                x_low_removed = remove_band(img_tensor, l_min, l_max, device)
                err_low_map = model.get_noise_pred_error(x_low_removed, t_step=T_STEP).squeeze(0).mean(dim=0).cpu().numpy()

                _, h_min, h_max = TARGET_HIGH
                x_high_removed = remove_band(img_tensor, h_min, h_max, device)
                err_high_map = model.get_noise_pred_error(x_high_removed, t_step=T_STEP).squeeze(0).mean(dim=0).cpu().numpy()

                diff_map = err_low_map - err_high_map

                if sum_fake_diff is None:
                    sum_fake_diff = np.zeros_like(diff_map)
                sum_fake_diff += diff_map
                fake_count += 1
            except Exception as e:
                continue
            
            if (i+1) % 20 == 0: print(f"  - Fake 누적 진행률: {i+1}/{num_pairs}")

    # --- 평균 계산 ---
    avg_real_diff = sum_real_diff / max(real_count, 1)
    avg_fake_diff = sum_fake_diff / max(fake_count, 1)

    # ==========================================
    # 📈 시각화: 지문(Fingerprint) 비교
    # ==========================================
    print("\n📊 평균 오차 맵(Fingerprint) 시각화 생성 중...")
    
    # 두 맵을 동일한 스케일로 비교하기 위해 절대값 최댓값 추출
    diff_abs_max = max(abs(avg_real_diff.min()), abs(avg_real_diff.max()), 
                       abs(avg_fake_diff.min()), abs(avg_fake_diff.max()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(avg_real_diff, cmap='seismic', vmin=-diff_abs_max, vmax=diff_abs_max)
    axes[0].set_title(f"REAL Average Diff Map (N={real_count})", fontsize=14, fontweight='bold', color='steelblue')
    axes[0].axis('off')

    im1 = axes[1].imshow(avg_fake_diff, cmap='seismic', vmin=-diff_abs_max, vmax=diff_abs_max)
    axes[1].set_title(f"FAKE Average Diff Map (N={fake_count})\n[Nano Banana Fingerprint]", fontsize=14, fontweight='bold', color='tomato')
    axes[1].axis('off')

    # 공통 컬러바
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label='Mean Diff Value (Low - High)')

    plt.suptitle(f"[Phase 5] Ensemble Average Error Map (Spatial Fingerprint)", fontsize=18, fontweight='bold')
    plt.subplots_adjust(right=0.9)
    save_path = os.path.join(SAVE_DIR, "ensemble_average_fingerprint.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"✅ Phase 5 분석 완료! '{save_path}' 결과물을 확인하세요.")

# --- 실행부 ---
MY_CHECKPOINT = "/data1/checkpoints/HYRE/past_mymodel5_bin_ep66_acc0.8856.pth"

# 100장 이상으로 세팅하는 것을 권장합니다.
run_phase5_ensemble_average(
    real_dir="/data1/DeepFake/REAL/DIV2K",
    fake_dir="/data1/DeepFake/FAKE/nano_banana",
    checkpoint_path=MY_CHECKPOINT,
    num_samples=NUM_SAMPLES,
)