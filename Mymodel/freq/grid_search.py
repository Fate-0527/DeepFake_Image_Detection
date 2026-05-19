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
NUM_SAMPLES = 100 # 테스트할 샘플 개수
T_STEP = Config.T_STEP  

LOW_CANDIDATES = [
    ("Low(30-130)", 30, 130),  # (비교군 1)
    ("Low(30-140)", 30, 140),  # (이전 최고점 부근)
    ("Low(30-150)", 30, 150),
    ("Low(30-160)", 30, 160),
    ("Low(30-170)", 30, 170),  # 512 이미지 기준 거의 모든 저/중주파 삭제
]

HIGH_CANDIDATES = [
    # 이전 실험에서 가장 타격감이 좋았던 최상위권 High 대역들로 압축
    ("High(160-210)", 160, 210),
    ("High(160-220)", 160, 220),
    ("High(170-210)", 170, 210),
    ("High(170-220)", 170, 220),
    ("High(170-230)", 170, 230),
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

def run_grid_search(real_dir, fake_dir, checkpoint_path=None, num_samples=20):
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

    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    combinations = list(itertools.product(LOW_CANDIDATES, HIGH_CANDIDATES))
    total_combos = len(combinations)
    print(f"📊 탐색 시작: 총 {total_combos}개의 조합 테스트 (샘플: {num_samples}개씩)")
    print("-" * 60)

    results = []

    for idx, ((l_name, l_min, l_max), (h_name, h_min, h_max)) in enumerate(combinations):
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
        score = np.abs(avg_real - avg_fake)

        results.append({
            "combo_name": f"{l_name}\nvs\n{h_name}", # 그래프 출력을 위해 줄바꿈 추가
            "score": score,
            "real_gap": avg_real,
            "fake_gap": avg_fake
        })
        print(f"[{idx+1}/{total_combos}] {l_name} vs {h_name} -> Score: {score:.4f} (Real: {avg_real:.4f} | Fake: {avg_fake:.4f})")

    # 분리도(Score) 기준으로 내림차순 정렬
    results.sort(key=lambda x: x['score'], reverse=True)

    # ==========================================
    # 📈 결과 시각화 (그래프 그리기) 추가
    # ==========================================
    # 상위 10개 조합만 그래프에 표시 (너무 많으면 글씨가 겹침)
    top_n = min(10, len(results))
    top_results = results[:top_n]
    
    combo_names = [r["combo_name"] for r in top_results]
    scores = [r["score"] for r in top_results]
    real_gaps = [r["real_gap"] for r in top_results]
    fake_gaps = [r["fake_gap"] for r in top_results]

    x = np.arange(len(combo_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 1. 분리도(Score) 꺾은선 그래프
    color = 'tab:red'
    ax1.set_xlabel('Frequency Band Combinations (Top {})'.format(top_n), fontsize=12, fontweight='bold')
    ax1.set_ylabel('Separation Score (Real Gap - Fake Gap)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(x, scores, color=color, marker='o', linewidth=2, markersize=8, label='Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(combo_names, rotation=0, fontsize=10)

    # 2. Real Gap vs Fake Gap 막대 그래프 (보조 축 사용)
    ax2 = ax1.twinx()  
    color_real = 'tab:blue'
    color_fake = 'tab:orange'
    ax2.set_ylabel('Mean Error Gap', fontsize=12, fontweight='bold')
    
    rects1 = ax2.bar(x - width/2, real_gaps, width, label='Real Gap', color=color_real, alpha=0.7)
    rects2 = ax2.bar(x + width/2, fake_gaps, width, label='Fake Gap', color=color_fake, alpha=0.7)

    # 막대 위에 수치 표시
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # 범례 합치기
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title('Grid Search Results: Best Frequency Band Combinations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = "high_compare_30-130_full_grid_search_results.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n🚀 시각화 완료! 결과 그래프 저장됨: {save_path}")
    plt.close(fig)

# --- 실행부 ---
MY_CHECKPOINT = "/data1/checkpoints/HYRE/past_mymodel5_bin_ep66_acc0.8856.pth"

run_grid_search(
    real_dir="/data1/DeepFake/REAL/DIV2K",
    fake_dir="/data1/DeepFake/FAKE/nano_banana",
    checkpoint_path=MY_CHECKPOINT,
    num_samples=NUM_SAMPLES,
)