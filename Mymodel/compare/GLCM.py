import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import glob
from skimage.feature import graycomatrix, graycoprops

# 사용자 모듈 임포트
from fire_model_binary import FIRE_model
from config import Config

# --- 설정 ---
NUM_SAMPLES = 50  # 분석하고 개별 저장할 이미지 쌍의 개수
T_STEP = Config.T_STEP
SAVE_DIR = "phase2_glcm_individual"

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

def normalize_for_glcm(diff_map, levels=32):
    """GLCM 계산을 위해 연속적인 실수 값을 0~(levels-1) 사이의 정수로 양자화"""
    d_min, d_max = diff_map.min(), diff_map.max()
    if d_max - d_min == 0:
        return np.zeros_like(diff_map, dtype=np.uint8)
    norm = (diff_map - d_min) / (d_max - d_min)
    quantized = np.clip(np.floor(norm * levels), 0, levels - 1).astype(np.uint8)
    return quantized

def extract_glcm_features(quantized_img, levels=32):
    """GLCM 생성 및 5가지 주요 텍스처 통계량 추출"""
    glcm = graycomatrix(quantized_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=levels, symmetric=True, normed=True)
    features = {
        'Contrast': graycoprops(glcm, 'contrast').mean(),           # 대조도
        'Dissimilarity': graycoprops(glcm, 'dissimilarity').mean(), # 비유사성
        'Homogeneity': graycoprops(glcm, 'homogeneity').mean(),     # 균일성
        'Energy': graycoprops(glcm, 'energy').mean(),               # 에너지
        'Correlation': graycoprops(glcm, 'correlation').mean()      # 상관관계
    }
    return features

def get_error_maps_and_glcm(img_path, model, device, t_step, low_band, high_band):
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

        diff_map = err_low_map - err_high_map # (Low - High) 기준
        
        # 정수로 양자화 후 GLCM 추출
        quantized_diff = normalize_for_glcm(diff_map, levels=32)
        glcm_features = extract_glcm_features(quantized_diff, levels=32)

    return {
        "orig": orig_img_np,
        "diff_map": diff_map,
        "glcm_features": glcm_features,
        "path": img_path
    }

def run_phase2_individual_analysis(real_dir, fake_dir, checkpoint_path, num_samples=50):
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

    print(f"📸 Phase 2: 개별 이미지 GLCM 텍스처 분석 시작 (총 {num_pairs}쌍)")
    feature_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation']

    for i in range(num_pairs):
        real_data = get_error_maps_and_glcm(real_files[i], model, device, T_STEP, TARGET_LOW, TARGET_HIGH)
        fake_data = get_error_maps_and_glcm(fake_files[i], model, device, T_STEP, TARGET_LOW, TARGET_HIGH)
        if real_data is None or fake_data is None: continue

        diff_abs_max = max(abs(real_data["diff_map"].min()), abs(real_data["diff_map"].max()), 
                           abs(fake_data["diff_map"].min()), abs(fake_data["diff_map"].max()))

        # 시각화 레이아웃
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 4)

        axes_img_r = fig.add_subplot(gs[0, 0])
        axes_diff_r = fig.add_subplot(gs[0, 1])
        axes_img_f = fig.add_subplot(gs[1, 0])
        axes_diff_f = fig.add_subplot(gs[1, 1])
        axes_bar = fig.add_subplot(gs[:, 2:]) # 우측 통계 그래프

        # --- Real ---
        axes_img_r.imshow(real_data["orig"])
        axes_img_r.set_ylabel("REAL", fontsize=16, fontweight='bold', color='steelblue')
        axes_img_r.set_title("Original Image")
        axes_img_r.set_xticks([]); axes_img_r.set_yticks([])

        axes_diff_r.imshow(real_data["diff_map"], cmap='seismic', vmin=-diff_abs_max, vmax=diff_abs_max)
        axes_diff_r.set_title("Diff Map (Low - High)")
        axes_diff_r.set_xticks([]); axes_diff_r.set_yticks([])

        # --- Fake ---
        axes_img_f.imshow(fake_data["orig"])
        axes_img_f.set_ylabel("FAKE", fontsize=16, fontweight='bold', color='tomato')
        axes_img_f.set_xticks([]); axes_img_f.set_yticks([])

        axes_diff_f.imshow(fake_data["diff_map"], cmap='seismic', vmin=-diff_abs_max, vmax=diff_abs_max)
        axes_diff_f.set_xticks([]); axes_diff_f.set_yticks([])

        # --- GLCM 막대 그래프 ---
        r_vals = [real_data["glcm_features"][k] for k in feature_names]
        f_vals = [fake_data["glcm_features"][k] for k in feature_names]
        
        x = np.arange(len(feature_names))
        width = 0.35

        # 시각적 스케일링을 위해 각 지표의 최대값 기준으로 0~1 정규화 (실제 값은 텍스트로 표기)
        max_vals = [max(r, f) for r, f in zip(r_vals, f_vals)]
        r_scaled = [r/m if m!=0 else 0 for r, m in zip(r_vals, max_vals)]
        f_scaled = [f/m if m!=0 else 0 for f, m in zip(f_vals, max_vals)]

        axes_bar.bar(x - width/2, r_scaled, width, label='Real Diff Map', color='steelblue', edgecolor='black')
        axes_bar.bar(x + width/2, f_scaled, width, label='Fake Diff Map', color='tomato', edgecolor='black')

        axes_bar.set_ylabel('Relative Feature Intensity', fontsize=12, fontweight='bold')
        axes_bar.set_title("GLCM Texture Statistics Comparison", fontsize=14, fontweight='bold')
        axes_bar.set_xticks(x)
        axes_bar.set_xticklabels(feature_names, fontsize=11, fontweight='bold')
        axes_bar.legend(fontsize=12, loc='center right')

        # 막대 위에 실제 수치 표기 (수치 직접 비교를 위함)
        for j, (r_val, f_val, r_s, f_s) in enumerate(zip(r_vals, f_vals, r_scaled, f_scaled)):
            axes_bar.text(x[j] - width/2, r_s + 0.02, f'{r_val:.3f}', ha='center', fontsize=11, fontweight='bold', color='steelblue')
            axes_bar.text(x[j] + width/2, f_s + 0.02, f'{f_val:.3f}', ha='center', fontsize=11, fontweight='bold', color='tomato')

        plt.suptitle(f"[Phase 2] Individual GLCM Analysis | Pair #{i+1}\nReal: {os.path.basename(real_data['path'])} vs Fake: {os.path.basename(fake_data['path'])}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_name = os.path.join(SAVE_DIR, f"glcm_pair_{i+1:02d}.png")
        plt.savefig(save_name, dpi=150)
        plt.close(fig)
        
        if (i+1) % 10 == 0:
            print(f"  - {i+1}개 이미지 저장 완료...")

    print(f"\n✅ 50쌍 개별 시각화 완료! '{SAVE_DIR}' 폴더에 저장되었습니다.")

# --- 실행부 ---
MY_CHECKPOINT = "/data1/checkpoints/HYRE/past_mymodel5_bin_ep66_acc0.8856.pth"

run_phase2_individual_analysis(
    real_dir="/data1/DeepFake/REAL/DIV2K",
    fake_dir="/data1/DeepFake/FAKE/nano_banana",
    checkpoint_path=MY_CHECKPOINT,
    num_samples=NUM_SAMPLES,
)