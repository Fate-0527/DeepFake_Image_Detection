import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import glob

# --- 설정 ---
NUM_SAMPLES = 100  # 군집을 확실히 보기 위해 모델당 200장씩 추출
SAVE_DIR = "idea1_scatter_plot"

# 타겟 대역폭 설정
R_LOW_MIN, R_LOW_MAX = 30, 110
R_HIGH_MIN, R_HIGH_MAX = 340, 350

def compute_band_energy(img_tensor):
    """이미지의 FFT 스펙트럼에서 특정 주파수 대역의 평균 에너지를 추출합니다."""
    # 1. 2D FFT 변환 및 로그 스케일링
    orig_np = img_tensor.squeeze(0).mean(dim=0).numpy() # 흑백화
    f_orig = np.fft.fft2(orig_np)
    f_shift = np.fft.fftshift(f_orig)
    magnitude = np.log(np.abs(f_shift) + 1e-8)
    
    # 2. 중심으로부터의 거리(r) 맵 생성
    center = (magnitude.shape[0] // 2, magnitude.shape[1] // 2)
    y, x = np.indices(magnitude.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # 3. 대역별 마스크 생성 및 평균 에너지 계산
    mask_low = (r >= R_LOW_MIN) & (r < R_LOW_MAX)
    mask_high = (r >= R_HIGH_MIN) & (r < R_HIGH_MAX)
    
    energy_low = np.mean(magnitude[mask_low])
    energy_high = np.mean(magnitude[mask_high])
    
    return energy_low, energy_high

def run_scatter_plot_analysis(real_dir, old_fake_dir, new_fake_dir, num_samples=200):
    os.makedirs(SAVE_DIR, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')

    print("=" * 60)
    print(f"🔬 [Idea 1] 세대별 주파수 에너지 산점도(Scatter Plot) 추출")
    print(f"   - X축: Low/Mid 대역 (r={R_LOW_MIN}~{R_LOW_MAX})")
    print(f"   - Y축: Extreme High 대역 (r={R_HIGH_MIN}~{R_HIGH_MAX})")
    print("=" * 60)

    # 데이터 추출용 헬퍼 함수
    def extract_energies(directory, label):
        files = sorted([f for f in glob.glob(os.path.join(directory, "*.*")) if f.lower().endswith(valid_exts)])[:num_samples]
        x_vals, y_vals = [], []
        print(f"▶ [{label}] 데이터 추출 중... (목표: {len(files)}장)")
        for fpath in files:
            try:
                img = transform(Image.open(fpath).convert('RGB'))
                e_low, e_high = compute_band_energy(img)
                x_vals.append(e_low)
                y_vals.append(e_high)
            except Exception: continue
        return x_vals, y_vals

    # 에너지 추출
    real_x, real_y = extract_energies(real_dir, "REAL")
    old_x, old_y = extract_energies(old_fake_dir, "OLD FAKE (SD 1.4)")
    new_x, new_y = extract_energies(new_fake_dir, "NEW FAKE (Nano Banana)")

    # ==========================================
    # 📈 시각화: Scatter Plot
    # ==========================================
    print("\n📊 산점도 시각화 생성 중...")
    plt.figure(figsize=(12, 10))

    # 점 찍기 (투명도 alpha를 주어 겹치는 부분 확인)
    plt.scatter(real_x, real_y, alpha=0.6, c='blue', edgecolors='k', s=50, label='REAL (True Photos)')
    plt.scatter(old_x, old_y, alpha=0.6, c='tomato', edgecolors='k', s=50, marker='s', label='OLD FAKE (SD 1.4)')
    plt.scatter(new_x, new_y, alpha=0.6, c='limegreen', edgecolors='k', s=50, marker='^', label='NEW FAKE (Nano Banana)')

    # 그래프 꾸미기
    plt.title("Generational Frequency Fingerprint of AI Models", fontsize=18, fontweight='bold')
    plt.xlabel(f"Low/Mid Frequency Energy (r={R_LOW_MIN}~{R_LOW_MAX})", fontsize=14)
    plt.ylabel(f"Extreme High Frequency Energy (r={R_HIGH_MIN}~{R_HIGH_MAX})", fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 군집 중심점(Mean) 표시
    def plot_center(x, y, color):
        plt.scatter(np.mean(x), np.mean(y), c=color, s=300, marker='X', edgecolors='black', zorder=5)
        
    plot_center(real_x, real_y, 'blue')
    plot_center(old_x, old_y, 'tomato')
    plot_center(new_x, new_y, 'limegreen')

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "340_350_generational_scatter_plot.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"✅ 추출 완료! '{save_path}'를 확인하여 세 그룹이 어떻게 갈라지는지 확인하세요.")

# --- 실행부 ---
run_scatter_plot_analysis(
    real_dir="/data1/DeepFake/train2017",
    old_fake_dir="/data1/DeepFake/FAKE/SD1_4",
    new_fake_dir="/data1/DeepFake/FAKE/nano_banana",
    num_samples=NUM_SAMPLES)