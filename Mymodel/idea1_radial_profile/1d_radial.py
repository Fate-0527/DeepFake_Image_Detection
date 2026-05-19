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
NUM_SAMPLES = 1000  # 각 클래스/모델당 분석할 이미지 수
T_STEP = Config.T_STEP
SAVE_DIR = "idea1_radial_profile"

def get_radial_profile(data, center):
    """2D 배열의 중심으로부터 거리(r)에 따른 1D 평균 프로파일을 계산합니다."""
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    
    # 거리(r)별로 값들을 합산하고 개수로 나누어 평균 계산
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)  # 0으로 나누기 방지
    return radialprofile

def compute_fft_radial_profiles(img_tensor, model, device, t_step):
    """원본 이미지의 FFT와 FIRE 모델 Raw 오차 맵의 FFT에 대한 방사형 프로파일을 구합니다."""
    # 1. 원본 이미지 FFT
    orig_np = img_tensor.squeeze(0).mean(dim=0).cpu().numpy() # [H, W] 평균 (흑백화)
    f_orig = np.fft.fft2(orig_np)
    f_orig_shift = np.fft.fftshift(f_orig)
    mag_orig = np.log(np.abs(f_orig_shift) + 1e-8)
    
    center = (mag_orig.shape[0] // 2, mag_orig.shape[1] // 2)
    radial_orig = get_radial_profile(mag_orig, center)

    # 2. Raw Error Map (어떤 주파수도 지우지 않은 순수 복원 오차) 계산
    with torch.no_grad():
        raw_error = model.get_noise_pred_error(img_tensor, t_step=t_step)
        err_np = raw_error.squeeze(0).mean(dim=0).cpu().numpy()
        
    f_err = np.fft.fft2(err_np)
    f_err_shift = np.fft.fftshift(f_err)
    mag_err = np.log(np.abs(f_err_shift) + 1e-8)
    
    radial_err = get_radial_profile(mag_err, center)
    
    return radial_orig, radial_err

def run_idea1_radial_analysis(real_dir, fake_base_dir, checkpoint_path, num_samples=50):
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
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
    ])
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')

    print("=" * 60)
    print(f"🔬 [Idea 1] 1D 방사형 주파수 프로파일 분석 시작")
    print("=" * 60)

    profiles = {'REAL': {'orig': [], 'err': []}}
    
    # 1. REAL 데이터 프로파일 누적
    real_files = sorted([f for f in glob.glob(os.path.join(real_dir, "*.*")) if f.lower().endswith(valid_exts)])[:num_samples]
    print(f"▶ [REAL] 분석 중... ({len(real_files)}장)")
    for fpath in real_files:
        try:
            img = transform(Image.open(fpath).convert('RGB')).unsqueeze(0).to(device)
            r_orig, r_err = compute_fft_radial_profiles(img, model, device, T_STEP)
            profiles['REAL']['orig'].append(r_orig)
            profiles['REAL']['err'].append(r_err)
        except Exception: continue

    # 2. FAKE 모델별 데이터 프로파일 누적
    fake_folders = sorted([f for f in os.listdir(fake_base_dir) if os.path.isdir(os.path.join(fake_base_dir, f))])
    
    for folder in fake_folders:
        profiles[folder] = {'orig': [], 'err': []}
        fake_files = sorted([f for f in glob.glob(os.path.join(fake_base_dir, folder, "*.*")) if f.lower().endswith(valid_exts)])[:num_samples]
        print(f"▶ [{folder}] 분석 중... ({len(fake_files)}장)")
        for fpath in fake_files:
            try:
                img = transform(Image.open(fpath).convert('RGB')).unsqueeze(0).to(device)
                r_orig, r_err = compute_fft_radial_profiles(img, model, device, T_STEP)
                profiles[folder]['orig'].append(r_orig)
                profiles[folder]['err'].append(r_err)
            except Exception: continue

    # ==========================================
    # 📈 시각화: 1D Radial Profile 꺾은선 그래프
    # ==========================================
    print("\n📊 프로파일 시각화 차트 생성 중...")
    
    # 수정된 부분: 1개의 플롯만 생성하도록 (1, 1)로 변경 및 비율 조정
    fig, ax = plt.subplots(figsize=(10, 7))

    # 최대 주파수 반경 (배열 길이) 통일
    max_len = min([len(p) for k in profiles for p in profiles[k]['orig']])
    x_axis = np.arange(max_len)

    colors = plt.cm.tab10(np.linspace(0, 1, len(fake_folders)))

    # (1) Original Image FFT Profile
    real_orig_mean = np.mean([p[:max_len] for p in profiles['REAL']['orig']], axis=0)
    ax.plot(x_axis, real_orig_mean, label='REAL', color='blue', linewidth=4, zorder=10)

    for idx, folder in enumerate(fake_folders):
        fake_orig_mean = np.mean([p[:max_len] for p in profiles[folder]['orig']], axis=0)
        ax.plot(x_axis, fake_orig_mean, label=f'Fake: {folder}', color=colors[idx], linewidth=1.5, alpha=0.8)

    ax.set_title("1D Radial Profile: Original Image FFT", fontsize=16, fontweight='bold')
    ax.set_xlabel("Frequency Radius (r)", fontsize=12)
    ax.set_ylabel("Log Magnitude", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Universal Frequency Vulnerability Search (Idea 1)", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, "try3_1000_1D_radial_profile_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"✅ Idea 1 완료! '{save_path}'를 확인하여 가장 간격이 넓은 X축(주파수 반경 r) 범위를 찾아보세요.")

# --- 실행부 ---
MY_CHECKPOINT = "/data1/checkpoints/HYRE/past_mymodel5_bin_ep66_acc0.8856.pth"

run_idea1_radial_analysis(
    real_dir="/data1/DeepFake/train2017",
    fake_base_dir="/data1/DeepFake/FAKE",
    checkpoint_path=MY_CHECKPOINT,
    num_samples=NUM_SAMPLES,
)