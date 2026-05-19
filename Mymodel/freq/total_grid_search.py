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
NUM_SAMPLES_PER_FAKE = 20 # 6개 모델 x 20개 = 총 120개 Fake 샘플
T_STEP = Config.T_STEP  

# 🚨 [수정됨] 범용성 확보를 위해 대역폭과 주파수 위치를 낮춘(Mid-Range) 새로운 후보군
LOW_CANDIDATES = [
    ("Low(20-70)", 20, 70),
    ("Low(30-80)", 30, 80),
    ("Low(40-90)", 40, 90),
    ("Low(50-100)", 50, 100),
    ("Low(50-120)", 50, 120),
]

HIGH_CANDIDATES = [
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

def run_generalized_search_and_validation(real_dir, fake_base_dir, checkpoint_path):
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
    print("📂 데이터 로딩 중...")
    
    # 생성 모델별 파일 경로 분리 저장 (Step 2를 위해)
    fake_folders = sorted([f for f in os.listdir(fake_base_dir) if os.path.isdir(os.path.join(fake_base_dir, f))])
    fake_files_by_model = {}
    total_fake_count = 0
    
    for folder in fake_folders:
        folder_path = os.path.join(fake_base_dir, folder)
        files = sorted([f for f in glob.glob(os.path.join(folder_path, "*.*")) if f.lower().endswith(valid_exts)])[:NUM_SAMPLES_PER_FAKE]
        fake_files_by_model[folder] = files
        total_fake_count += len(files)
        print(f"  - [{folder}] 샘플 {len(files)}개 로드 완료")

    all_real_files = sorted([f for f in glob.glob(os.path.join(real_dir, "*.*")) if f.lower().endswith(valid_exts)])
    real_files = all_real_files[:total_fake_count]
    
    combinations = list(itertools.product(LOW_CANDIDATES, HIGH_CANDIDATES))
    total_combos = len(combinations)
    
    print("-" * 60)
    print(f"🏆 Step 1 & 2 시작: 총 {total_combos}개 조합 교차 검증 (Real {len(real_files)} vs Mixed Fake {total_fake_count})")
    print("-" * 60)

    results = []

    for idx, ((l_name, l_min, l_max), (h_name, h_min, h_max)) in enumerate(combinations):
        real_gaps = []
        # Real 평가
        for fpath in real_files:
            try:
                img_t = transform(Image.open(fpath).convert('RGB')).unsqueeze(0).to(device)
                gap = get_single_band_error(img_t, model, h_min, h_max, T_STEP, device) - get_single_band_error(img_t, model, l_min, l_max, T_STEP, device)
                real_gaps.append(gap)
            except Exception: continue

        avg_real = np.mean(real_gaps) if real_gaps else 0
        global_fake_gaps = []
        model_specific_scores = {}

        # 6개 Fake 모델별 평가 (한 번의 루프로 모델별/글로벌 점수 동시 획득)
        for folder, files in fake_files_by_model.items():
            model_fake_gaps = []
            for fpath in files:
                try:
                    img_t = transform(Image.open(fpath).convert('RGB')).unsqueeze(0).to(device)
                    gap = get_single_band_error(img_t, model, h_min, h_max, T_STEP, device) - get_single_band_error(img_t, model, l_min, l_max, T_STEP, device)
                    model_fake_gaps.append(gap)
                except Exception: continue
            
            global_fake_gaps.extend(model_fake_gaps)
            # 해당 생성 모델 전용 Score 계산
            avg_model_fake = np.mean(model_fake_gaps) if model_fake_gaps else 0
            model_specific_scores[folder] = avg_real - avg_model_fake

        avg_global_fake = np.mean(global_fake_gaps) if global_fake_gaps else 0
        global_score = avg_real - avg_global_fake

        results.append({
            "combo_name": f"{l_name}\nvs\n{h_name}",
            "global_score": global_score,
            "real_gap": avg_real,
            "fake_gap": avg_global_fake,
            "model_scores": model_specific_scores
        })
        print(f"[{idx+1}/{total_combos}] {l_name} vs {h_name} -> Global Score: {global_score:.4f}")

    # 글로벌 점수 기준으로 내림차순 정렬
    results.sort(key=lambda x: x['global_score'], reverse=True)
    top_n = min(10, len(results))
    top_results = results[:top_n]

    combo_names = [r["combo_name"] for r in top_results]
    global_scores = [r["global_score"] for r in top_results]

    # ==========================================
    # 📈 [시각화 1] 글로벌 범용성 Top 10 차트
    # ==========================================
    x = np.arange(len(combo_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(16, 8))
    color = 'tab:red'
    ax1.set_xlabel('Top 10 Generalized Frequency Band Combinations', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Global Score (Real Gap - Mixed Fake Gap)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(x, global_scores, color=color, marker='o', linewidth=3, markersize=10, label='Global Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(combo_names, rotation=0, fontsize=10)

    for i, txt in enumerate(global_scores):
        ax1.annotate(f"{txt:.4f}", (x[i], global_scores[i]), textcoords="offset points", xytext=(0,10), ha='center', color=color, fontweight='bold')

    ax2 = ax1.twinx()  
    color_real, color_fake = 'tab:blue', 'tab:orange'
    ax2.set_ylabel('Mean Error Gap', fontsize=12, fontweight='bold')
    ax2.bar(x - width/2, [r["real_gap"] for r in top_results], width, label='Real Gap', color=color_real, alpha=0.7)
    ax2.bar(x + width/2, [r["fake_gap"] for r in top_results], width, label='Fake Gap (Mixed)', color=color_fake, alpha=0.7)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title(f'Generalized Grid Search (Mid-Range Focus)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("step1_mid_range_global_scores.png", dpi=150)
    plt.close(fig)

    # ==========================================
    # 📈 [시각화 2] 6개 생성 모델별 성적표 (일반화 검증)
    # ==========================================
    plt.figure(figsize=(18, 8))
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # 모델별로 점수를 추출하여 꺾은선으로 그림
    for idx, folder in enumerate(fake_folders):
        scores_for_model = [r["model_scores"][folder] for r in top_results]
        plt.plot(combo_names, scores_for_model, marker=markers[idx % len(markers)], linewidth=2, markersize=8, label=folder)

    # 전체 평균(Global Score)도 굵은 점선으로 표시
    plt.plot(combo_names, global_scores, color='black', linestyle='--', linewidth=3, label='Global Average')

    plt.title('Separation Score across Top 10 Mid-Range Combinations by Generator Model', fontsize=16, fontweight='bold')
    plt.xlabel('Top 10 Frequency Band Combinations', fontsize=12, fontweight='bold')
    plt.ylabel('Score (Real Gap - Fake Gap)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0, fontsize=10)
    plt.legend(title='Generator Models', fontsize=11, title_fontsize=12, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("step2_model_by_model_validation.png", dpi=150)
    plt.close()

    print(f"\n✅ 파이프라인 완료! 결과 확인:")
    print(f"  1. 글로벌 평가 차트: step1_mid_range_global_scores.png")
    print(f"  2. 모델별 일반화 검증 차트: step2_model_by_model_validation.png")

# --- 실행부 ---
MY_CHECKPOINT = "/data1/checkpoints/HYRE/past_mymodel5_bin_ep66_acc0.8856.pth"

run_generalized_search_and_validation(
    real_dir="/data1/DeepFake/REAL/DIV2K",
    fake_base_dir="/data1/DeepFake/FAKE",
    checkpoint_path=MY_CHECKPOINT
)