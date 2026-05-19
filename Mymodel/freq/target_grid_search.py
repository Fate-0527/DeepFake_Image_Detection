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
# 🚨 샘플 수를 20에서 200(혹은 그 이상)으로 대폭 늘립니다.
# (데이터셋 폴더에 있는 이미지 수에 맞춰서 조절하세요. 많을수록 정확합니다.)
NUM_SAMPLES = 200 
T_STEP = Config.T_STEP  

# [수정] 결승전에 진출한 Top 10 정예 조합 리스트
# (표기용 이름, low_min, low_max, high_min, high_max)
TARGET_COMBINATIONS = [
    ("Low(30-140)\nvs\nHigh(170-220)", 30, 140, 170, 220),
    ("Low(30-130)\nvs\nHigh(170-230)", 30, 130, 170, 230),
    ("Low(30-140)\nvs\nHigh(160-210)", 30, 140, 160, 210),
    ("Low(30-150)\nvs\nHigh(170-230)", 30, 150, 170, 230),
    ("Low(30-120)\nvs\nHigh(170-220)", 30, 120, 170, 220),
    ("Low(30-160)\nvs\nHigh(170-220)", 30, 160, 170, 220),
    ("Low(30-130)\nvs\nHigh(170-210)", 30, 130, 170, 210),
    ("Low(30-130)\nvs\nHigh(160-210)", 30, 130, 160, 210),
    ("Low(30-150)\nvs\nHigh(170-210)", 30, 150, 170, 210),
    ("Low(30-120)\nvs\nHigh(160-220)", 30, 120, 160, 220),
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

def run_final_round(real_dir, fake_dir, checkpoint_path=None, num_samples=200):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FIRE_model(device=device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        print(f"✅ Checkpoint 로드 완료: {checkpoint_path}")
    model.eval()

    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    real_files = sorted([f for f in glob.glob(os.path.join(real_dir, "*.*")) if f.lower().endswith(valid_exts)])[:num_samples]
    fake_files = sorted([f for f in glob.glob(os.path.join(fake_dir, "*.*")) if f.lower().endswith(valid_exts)])[:num_samples]
    
    # 실제 가져온 샘플 수 확인
    actual_real = len(real_files)
    actual_fake = len(fake_files)

    print(f"🏆 Final Round 시작: 총 10개 조합 테스트 (샘플: Real {actual_real}개, Fake {actual_fake}개)")
    print("-" * 60)

    results = []
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    for idx, (name, l_min, l_max, h_min, h_max) in enumerate(TARGET_COMBINATIONS):
        real_gaps = []
        fake_gaps = []

        # Real 이미지 평가
        for fpath in real_files:
            try:
                img = Image.open(fpath).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(device)
                
                err_low = get_single_band_error(img_t, model, l_min, l_max, T_STEP, device)
                err_high = get_single_band_error(img_t, model, h_min, h_max, T_STEP, device)
                real_gaps.append(err_high - err_low)
            except Exception:
                continue

        # Fake 이미지 평가
        for fpath in fake_files:
            try:
                img = Image.open(fpath).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(device)
                
                err_low = get_single_band_error(img_t, model, l_min, l_max, T_STEP, device)
                err_high = get_single_band_error(img_t, model, h_min, h_max, T_STEP, device)
                fake_gaps.append(err_high - err_low)
            except Exception:
                continue

        if not real_gaps or not fake_gaps:
            continue

        avg_real = np.mean(real_gaps)
        avg_fake = np.mean(fake_gaps)
        score = avg_real - avg_fake

        results.append({
            "combo_name": name,
            "score": score,
            "real_gap": avg_real,
            "fake_gap": avg_fake
        })
        print(f"[{idx+1}/10] {name.replace('\n', ' ')} -> Score: {score:.4f}")

    # 점수 기준으로 내림차순 정렬 (최종 순위 발표)
    results.sort(key=lambda x: x['score'], reverse=True)

    # ==========================================
    # 📈 결과 시각화 (그래프 그리기)
    # ==========================================
    combo_names = [r["combo_name"] for r in results]
    scores = [r["score"] for r in results]
    real_gaps = [r["real_gap"] for r in results]
    fake_gaps = [r["fake_gap"] for r in results]

    x = np.arange(len(combo_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(15, 8))

    color = 'tab:red'
    ax1.set_xlabel('Top 10 Frequency Band Combinations', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Separation Score (Real - Fake)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(x, scores, color=color, marker='o', linewidth=3, markersize=10, label='Final Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(combo_names, rotation=0, fontsize=10)

    # 값 표시 (Score)
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

    plt.title(f'Final Round Results (N={actual_real} Real vs {actual_fake} Fake)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = "final_round_results.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n🚀 최종 결승전 완료! 결과 그래프 저장됨: {save_path}")
    plt.close(fig)

# --- 실행부 ---
MY_CHECKPOINT = "/data1/checkpoints/HYRE/past_mymodel5_bin_ep66_acc0.8856.pth"

run_final_round(
    real_dir="/data1/DeepFake/REAL/DIV2K",
    fake_dir="/data1/DeepFake/FAKE/nano_banana",
    checkpoint_path=MY_CHECKPOINT,
    num_samples=NUM_SAMPLES,
)