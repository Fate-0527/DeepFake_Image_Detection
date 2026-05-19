import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import glob
from scipy import ndimage

# 사용자 모듈 임포트
from fire_model_binary import FIRE_model
from config import Config

# --- 설정 ---
NUM_SAMPLES = 50  # 분석할 이미지 쌍의 개수
T_STEP = Config.T_STEP
SAVE_DIR = "phase3_gradient_analysis"

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

def compute_gradient_magnitude(diff_map):
    """Sobel 필터를 사용하여 Diff Map의 그래디언트(변화량) 강도를 계산합니다."""
    # x축, y축 방향의 편미분(Sobel)
    dx = ndimage.sobel(diff_map, axis=0)
    dy = ndimage.sobel(diff_map, axis=1)
    # 그래디언트 벡터의 크기(Magnitude)
    magnitude = np.hypot(dx, dy)
    return magnitude

def extract_gradient_features(grad_map):
    """그래디언트 맵에서 오차의 '날카로움'을 나타내는 통계량을 추출합니다."""
    return {
        'Mean': np.mean(grad_map),                   # 전반적인 변화량 평균
        'Variance': np.var(grad_map),                # 변화량이 얼마나 들쭉날쭉한지(분산)
        'Max_99%': np.percentile(grad_map, 99)       # 아웃라이어를 제외한 최상위 1%의 날카로운 에지 강도
    }

def get_error_maps_and_gradient(img_path, model, device, t_step, low_band, high_band):
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
    ])
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

    img_tensor = transform(img).unsqueeze(0).to(device)
    orig_img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    with torch.no_grad():
        _, l_min, l_max = low_band
        x_low_removed = remove_band(img_tensor, l_min, l_max, device)
        err_low_map = model.get_noise_pred_error(x_low_removed, t_step=t_step).squeeze(0).mean(dim=0).cpu().numpy()

        _, h_min, h_max = high_band
        x_high_removed = remove_band(img_tensor, h_min, h_max, device)
        err_high_map = model.get_noise_pred_error(x_high_removed, t_step=t_step).squeeze(0).mean(dim=0).cpu().numpy()

        # Diff Map 계산
        diff_map = err_low_map - err_high_map
        
        # 🌟 핵심: Diff Map에 Sobel 필터 적용하여 그래디언트 맵 생성
        grad_map = compute_gradient_magnitude(diff_map)
        grad_features = extract_gradient_features(grad_map)

    return {
        "orig": orig_img_np,
        "diff_map": diff_map,
        "grad_map": grad_map,
        "features": grad_features,
        "path": img_path
    }

def run_phase3_gradient_analysis(real_dir, fake_dir, checkpoint_path, num_samples=50):
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

    print(f"📸 Phase 3: 오차 맵 그래디언트(Sobel) 분석 시작 (총 {num_pairs}쌍)")
    feature_names = ['Mean', 'Variance', 'Max_99%']

    for i in range(num_pairs):
        real_data = get_error_maps_and_gradient(real_files[i], model, device, T_STEP, TARGET_LOW, TARGET_HIGH)
        fake_data = get_error_maps_and_gradient(fake_files[i], model, device, T_STEP, TARGET_LOW, TARGET_HIGH)
        if real_data is None or fake_data is None: continue

        # 시각화 스케일 통일
        diff_abs_max = max(abs(real_data["diff_map"].min()), abs(real_data["diff_map"].max()), 
                           abs(fake_data["diff_map"].min()), abs(fake_data["diff_map"].max()))
        grad_vmax = max(real_data["grad_map"].max(), fake_data["grad_map"].max()) * 0.8 # 상위 20%는 포화시켜 에지 강조

        # 시각화 레이아웃 설정
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 4)

        axes_img_r = fig.add_subplot(gs[0, 0])
        axes_diff_r = fig.add_subplot(gs[0, 1])
        axes_grad_r = fig.add_subplot(gs[0, 2])
        axes_img_f = fig.add_subplot(gs[1, 0])
        axes_diff_f = fig.add_subplot(gs[1, 1])
        axes_grad_f = fig.add_subplot(gs[1, 2])
        axes_bar = fig.add_subplot(gs[:, 3]) # 우측 통계 그래프

        # --- Real ---
        axes_img_r.imshow(real_data["orig"])
        axes_img_r.set_ylabel("REAL", fontsize=16, fontweight='bold', color='steelblue')
        axes_img_r.set_title("Original Image")
        axes_img_r.set_xticks([]); axes_img_r.set_yticks([])

        axes_diff_r.imshow(real_data["diff_map"], cmap='seismic', vmin=-diff_abs_max, vmax=diff_abs_max)
        axes_diff_r.set_title("Diff Map")
        axes_diff_r.set_xticks([]); axes_diff_r.set_yticks([])

        axes_grad_r.imshow(real_data["grad_map"], cmap='hot', vmin=0, vmax=grad_vmax)
        axes_grad_r.set_title("Gradient (Sobel) Map")
        axes_grad_r.set_xticks([]); axes_grad_r.set_yticks([])

        # --- Fake ---
        axes_img_f.imshow(fake_data["orig"])
        axes_img_f.set_ylabel("FAKE", fontsize=16, fontweight='bold', color='tomato')
        axes_img_f.set_xticks([]); axes_img_f.set_yticks([])

        axes_diff_f.imshow(fake_data["diff_map"], cmap='seismic', vmin=-diff_abs_max, vmax=diff_abs_max)
        axes_diff_f.set_xticks([]); axes_diff_f.set_yticks([])

        axes_grad_f.imshow(fake_data["grad_map"], cmap='hot', vmin=0, vmax=grad_vmax)
        axes_grad_f.set_xticks([]); axes_grad_f.set_yticks([])

        # --- 그래디언트 통계 막대 그래프 ---
        r_vals = [real_data["features"][k] for k in feature_names]
        f_vals = [fake_data["features"][k] for k in feature_names]
        
        x = np.arange(len(feature_names))
        width = 0.35

        # 스케일이 서로 다르므로 각각의 최대값 기준으로 정규화하여 표기
        max_vals = [max(r, f) for r, f in zip(r_vals, f_vals)]
        r_scaled = [r/m if m!=0 else 0 for r, m in zip(r_vals, max_vals)]
        f_scaled = [f/m if m!=0 else 0 for f, m in zip(f_vals, max_vals)]

        axes_bar.bar(x - width/2, r_scaled, width, label='Real Gradient', color='steelblue', edgecolor='black')
        axes_bar.bar(x + width/2, f_scaled, width, label='Fake Gradient', color='tomato', edgecolor='black')

        axes_bar.set_ylabel('Relative Intensity', fontsize=12, fontweight='bold')
        axes_bar.set_title("Gradient Sharpness Statistics", fontsize=14, fontweight='bold')
        axes_bar.set_xticks(x)
        axes_bar.set_xticklabels(feature_names, fontsize=11, fontweight='bold')
        axes_bar.legend(fontsize=11, loc='upper right')

        # 막대 위에 실제 수치 표기
        for j, (r_val, f_val, r_s, f_s) in enumerate(zip(r_vals, f_vals, r_scaled, f_scaled)):
            axes_bar.text(x[j] - width/2, r_s + 0.02, f'{r_val:.5f}', ha='center', fontsize=10, fontweight='bold', color='steelblue')
            axes_bar.text(x[j] + width/2, f_s + 0.02, f'{f_val:.5f}', ha='center', fontsize=10, fontweight='bold', color='tomato')

        plt.suptitle(f"[Phase 3] Edge & Gradient Analysis | Pair #{i+1}\nReal: {os.path.basename(real_data['path'])} vs Fake: {os.path.basename(fake_data['path'])}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_name = os.path.join(SAVE_DIR, f"gradient_pair_{i+1:02d}.png")
        plt.savefig(save_name, dpi=150)
        plt.close(fig)
        
        if (i+1) % 10 == 0:
            print(f"  - {i+1}개 이미지 저장 완료...")

    print(f"\n✅ Phase 3 그래디언트 시각화 완료! '{SAVE_DIR}' 폴더에 저장되었습니다.")

# --- 실행부 ---
MY_CHECKPOINT = "/data1/checkpoints/HYRE/past_mymodel5_bin_ep66_acc0.8856.pth"

# 터미널에서 실행 시 해당 경로 설정
run_phase3_gradient_analysis(
    real_dir="/data1/DeepFake/REAL/DIV2K",
    fake_dir="/data1/DeepFake/FAKE/nano_banana",
    checkpoint_path=MY_CHECKPOINT,
    num_samples=NUM_SAMPLES,
)