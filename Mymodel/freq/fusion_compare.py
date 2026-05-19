import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import glob
import itertools

# 사용자 모듈 임포트
from fire_model_binary import FIRE_model
from config import Config

# --- 설정 ---
NUM_SAMPLES_PER_FAKE = 20 # 6개 모델 x 20개 = 총 120개 Fake (속도를 위해 조절, 원하시면 늘려도 됩니다)
T_STEP = Config.T_STEP  

# [수정됨] 범용성 확보를 위해 대역대를 대폭 낮춘 새로운 탐색 후보군
LOW_CANDIDATES = [
    ("Low(20-80)", 20, 80),
    ("Low(30-90)", 30, 90),
    ("Low(30-100)", 30, 100),
    ("Low(30-110)", 30, 110),
    ("Low(40-100)", 40, 100),
    ("Low(40-120)", 40, 120),
]

HIGH_CANDIDATES = [
    ("High(80-130)", 80, 130),
    ("High(100-150)", 100, 150),
    ("High(120-170)", 120, 170),
    ("High(130-180)", 130, 180),
    ("High(140-190)", 140, 190),
    ("High(150-200)", 150, 200),
]

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

def get_single_band_error(img_tensor, model, r_low, r_high, t_step, device):
    with torch.no_grad():
        x_removed = remove_band(img_tensor, r_low, r_high, device)
        error = model.get_noise_pred_error(x_removed, t_step=t_step)
        return error.mean().item()

def run_generalized_grid_search(real_dir, fake_base_dir, checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FIRE_model(device=device)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        print(f"✅ Checkpoint 로드 완료: {checkpoint_path}")
    model.eval()

    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    print("-" * 60)
    print("📂 데이터 로딩 중 (모든 생성 모델 통합)...")
    
    # 여러 생성 모델에서 샘플 추출하여 하나의 Fake 집단으로 통합
    fake_folders = [f for f in os.listdir(fake_base_dir) if os.path.isdir(os.path.join(fake_base_dir, f))]
    fake_files = []
    
    for folder in fake_folders:
        folder_path = os.path.join(fake_base_dir, folder)
        files = sorted([f for f in glob.glob(os.path.join(folder_path, "*.*")) if f.lower().endswith(valid_exts)])
        sampled_files = files[:NUM_SAMPLES_PER_FAKE]
        fake_files.extend(sampled_files)
        print(f"  - [{folder}] 폴더에서 {len(sampled_files)}개 로드 완료")

    # Real 이미지 1:1 매칭
    all_real_files = sorted([f for f in glob.glob(os.path.join(real_dir, "*.*")) if f.lower().endswith(valid_exts)])
    real_files = all_real_files[:len(fake_files)]
    
    actual_real = len(real_files)
    actual_fake = len(fake_files)

    combinations = list(itertools.product(LOW_CANDIDATES, HIGH_CANDIDATES))
    total_combos = len(combinations)
    
    print("-" * 60)
    print(f"🏆 Generalized Grid Search 시작: 총 {total_combos}개 조합 테스트")
    print(f"📊 테스트 샘플: Real {actual_real}개 vs Mixed Fake {actual_fake}개")
    print("-" * 60)

    results = []

    for idx, ((l_name, l_min, l_max), (h_name, h_min, h_max)) in enumerate(combinations):
        real_gaps, fake_gaps = [], []

        for fpath in real_files:
            try:
                img_t = transform(Image.open(fpath).convert('RGB')).unsqueeze(0).to(device)
                err_low = get_single_band_error(img_t, model, l_min, l_max, T_STEP, device)
                err_high = get_single_band_error(img_t, model, h_min, h_max, T_STEP, device)
                real_gaps.append(err_high - err_low)
            except Exception: continue

        for fpath in fake_files:
            try:
                img_t = transform(Image.open(fpath).convert('RGB')).unsqueeze(0).to(device)
                err_low = get_single_band_error(img_t, model, l_min, l_max, T_STEP, device)
                err_high = get_single_band_error(img_t, model, h_min, h_max, T_STEP, device)
                fake_gaps.append(err_high - err_low)
            except Exception: continue

        if not real_gaps or not fake_gaps: continue

        avg_real = np.mean(real_gaps)
        avg_fake = np.mean(fake_gaps)
        score = avg_real - avg_fake

        results.append({
            "combo_name": f"{l_name}\nvs\n{h_name}",
            "score": score,
            "real_gap": avg_real,
            "fake_gap": avg_fake
        })
        print(f"[{idx+1}/{total_combos}] {l_name} vs {h_name} -> Score: {score:.4f} (R: {avg_real:.4f} | F: {avg_fake:.4f})")

    results.sort(key=lambda x: x['score'], reverse=True)

    # --- 📈 결과 시각화 ---
    top_n = min(10, len(results))
    top_results = results[:top_n]
    
    combo_names = [r["combo_name"] for r in top_results]
    scores = [r["score"] for r in top_results]
    real_gaps = [r["real_gap"] for r in top_results]
    fake_gaps = [r["fake_gap"] for r in top_results]

    x = np.arange(len(combo_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(16, 8))

    color = 'tab:red'
    ax1.set_xlabel(f'Top {top_n} Generalized Frequency Band Combinations', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Separation Score (Real Gap - Fake Gap)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(x, scores, color=color, marker='o', linewidth=3, markersize=10, label='Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(combo_names, rotation=0, fontsize=10)

    for i, txt in enumerate(scores):
        ax1.annotate(f"{txt:.4f}", (x[i], scores[i]), textcoords="offset points", xytext=(0,10), ha='center', color=color, fontweight='bold')

    ax2 = ax1.twinx()  
    color_real = 'tab:blue'
    color_fake = 'tab:orange'
    ax2.set_ylabel('Mean Error Gap', fontsize=12, fontweight='bold')
    
    rects1 = ax2.bar(x - width/2, real_gaps, width, label='Real Gap', color=color_real, alpha=0.7)
    rects2 = ax2.bar(x + width/2, fake_gaps, width, label='Fake Gap', color=color_fake, alpha=0.7)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title(f'Generalized Grid Search Results\n(N={actual_real} Real vs {actual_fake} Mixed Fake)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = "generalized_grid_search_results.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n🚀 탐색 완료! 최적의 범용 주파수 그래프 저장됨: {save_path}")
    plt.close(fig)

# --- 실행부 ---
MY_CHECKPOINT = "/data1/checkpoints/HYRE/past_mymodel5_bin_ep66_acc0.8856.pth"

run_generalized_grid_search(
    real_dir="/data1/DeepFake/REAL/DIV2K",
    fake_base_dir="/data1/DeepFake/FAKE",
    checkpoint_path=MY_CHECKPOINT
)