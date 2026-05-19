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
NUM_SAMPLES_PER_FAKE = 100 # 모델별 100개씩 추출
T_STEP = Config.T_STEP  

# [결승전 진출 Top 10 조합]
TARGET_COMBINATIONS = [
    ("Low(30-140) vs High(170-220)", 30, 140, 170, 220),
    ("Low(30-130) vs High(170-230)", 30, 130, 170, 230),
    ("Low(30-140) vs High(160-210)", 30, 140, 160, 210),
    ("Low(30-150) vs High(170-230)", 30, 150, 170, 230),
    ("Low(30-120) vs High(170-220)", 30, 120, 170, 220),
    ("Low(30-160) vs High(170-220)", 30, 160, 170, 220),
    ("Low(30-130) vs High(170-210)", 30, 130, 170, 210),
    ("Low(30-130) vs High(160-210)", 30, 130, 160, 210),
    ("Low(30-150) vs High(170-210)", 30, 150, 170, 210),
    ("Low(30-120) vs High(160-220)", 30, 120, 160, 220),
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
    img_filtered = torch.abs(torch.fft.ifftn(torch.fft.ifftshift(freq_filtered, dim=(-2, -1)), dim=(-2, -1)))
    flat = img_filtered.view(B, -1)
    norm = torch.norm(flat, p=2, dim=1).view(B, 1, 1, 1)
    return img_filtered / (norm + 1e-8)

def get_single_band_error(img_tensor, model, r_low, r_high, t_step, device):
    with torch.no_grad():
        x_removed = remove_band(img_tensor, r_low, r_high, device)
        error = model.get_noise_pred_error(x_removed, t_step=t_step)
        return error.mean().item()

def run_model_by_model_evaluation(real_dir, fake_base_dir, checkpoint_path):
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
    transform = transforms.Compose([transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)), transforms.ToTensor()])

    fake_folders = sorted([f for f in os.listdir(fake_base_dir) if os.path.isdir(os.path.join(fake_base_dir, f))])
    all_real_files = sorted([f for f in glob.glob(os.path.join(real_dir, "*.*")) if f.lower().endswith(valid_exts)])
    
    # 모델별 점수를 저장할 딕셔너리 { "nano_banana": [score1, score2...], "SD1_4": [...] }
    model_scores = {folder: [] for folder in fake_folders}
    combo_names = [c[0] for c in TARGET_COMBINATIONS]

    print(f"\n🚀 Phase 1: 생성 모델별 Score(Gap) 정량 평가 시작 (각 100개씩)")
    print("-" * 70)

    for folder in fake_folders:
        folder_path = os.path.join(fake_base_dir, folder)
        fake_files = sorted([f for f in glob.glob(os.path.join(folder_path, "*.*")) if f.lower().endswith(valid_exts)])[:NUM_SAMPLES_PER_FAKE]
        real_files = all_real_files[:len(fake_files)] # 공평하게 1:1 매칭
        
        print(f"\n🔍 평가 중: [{folder}] (Real {len(real_files)} vs Fake {len(fake_files)})")
        
        for name, l_min, l_max, h_min, h_max in TARGET_COMBINATIONS:
            real_gaps, fake_gaps = [], []
            
            for fpath in real_files:
                try:
                    img_t = transform(Image.open(fpath).convert('RGB')).unsqueeze(0).to(device)
                    real_gaps.append(get_single_band_error(img_t, model, h_min, h_max, T_STEP, device) - get_single_band_error(img_t, model, l_min, l_max, T_STEP, device))
                except Exception: continue
                
            for fpath in fake_files:
                try:
                    img_t = transform(Image.open(fpath).convert('RGB')).unsqueeze(0).to(device)
                    fake_gaps.append(get_single_band_error(img_t, model, h_min, h_max, T_STEP, device) - get_single_band_error(img_t, model, l_min, l_max, T_STEP, device))
                except Exception: continue

            if real_gaps and fake_gaps:
                score = np.mean(real_gaps) - np.mean(fake_gaps)
                model_scores[folder].append(score)
                print(f"  - {name}: Score = {score:.4f}")

    # --- 📈 시각화 (모델별 성능 꺾은선 그래프) ---
    plt.figure(figsize=(16, 8))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    
    for idx, (folder, scores) in enumerate(model_scores.items()):
        plt.plot(combo_names, scores, marker=markers[idx % len(markers)], linewidth=2, markersize=8, label=folder)

    plt.title('Separation Score across Top 10 Combinations by Generator Model', fontsize=16, fontweight='bold')
    plt.xlabel('Top 10 Frequency Band Combinations', fontsize=12, fontweight='bold')
    plt.ylabel('Score (Real Gap - Fake Gap)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.legend(title='Generator Models', fontsize=10, title_fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = "model_by_model_scores.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Phase 1 완료! 그래프 저장됨: {save_path}")

# --- 실행부 ---
MY_CHECKPOINT = "/data1/checkpoints/HYRE/past_mymodel5_bin_ep66_acc0.8856.pth"
run_model_by_model_evaluation(
    real_dir="/data1/DeepFake/REAL/DIV2K",
    fake_base_dir="/data1/DeepFake/FAKE",
    checkpoint_path=MY_CHECKPOINT
)