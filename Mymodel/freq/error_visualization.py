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
NUM_SAMPLES = 50
T_STEP = Config.T_STEP
SAVE_DIR = "error_pattern_reversed"

# 분석할 대역 조합
TARGET_LOW = ("Low(30-140)", 30, 140)
TARGET_HIGH = ("High(170-220)", 170, 220)

def make_band_mask(rows, cols, r_low, r_high, device):
    crow, ccol = rows // 2, cols // 2
    y, x = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')
    dist_sq = (x - ccol) ** 2 + (y - crow) ** 2
    mask = ((dist_sq >= r_low**2) & (dist_sq < r_high**2)).float()
    return mask.unsqueeze(0).unsqueeze(0).to(device)

def calc_band_energy(img_tensor, r_low, r_high, device):
    _, _, H, W = img_tensor.shape
    freq = torch.fft.fftn(img_tensor, dim=(-2, -1))
    freq = torch.fft.fftshift(freq, dim=(-2, -1))
    band_mask = make_band_mask(H, W, r_low, r_high, device)
    mag = torch.abs(freq)
    active_bins = torch.sum(band_mask)
    if active_bins == 0:
        return 1e-5
    band_energy = torch.sum(mag * band_mask) / active_bins
    return band_energy.item() + 1e-8

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

def get_normalized_error_maps(img_path, model, device, t_step, low_band, high_band):
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
        energy_low = calc_band_energy(img_tensor, l_min, l_max, device)
        x_low_removed = remove_band(img_tensor, l_min, l_max, device)
        error_low = model.get_noise_pred_error(x_low_removed, t_step=t_step)
        
        err_low_map = error_low.squeeze(0).mean(dim=0).cpu().numpy()
        norm_err_low_map = err_low_map / energy_low

        _, h_min, h_max = high_band
        energy_high = calc_band_energy(img_tensor, h_min, h_max, device)
        x_high_removed = remove_band(img_tensor, h_min, h_max, device)
        error_high = model.get_noise_pred_error(x_high_removed, t_step=t_step)
        
        err_high_map = error_high.squeeze(0).mean(dim=0).cpu().numpy()
        norm_err_high_map = err_high_map / energy_high

        # 🚨 [수정됨] 점수 계산 방식을 완전히 반대로(Low - High) 뒤집음
        raw_diff_map = err_low_map - err_high_map
        norm_diff_map = norm_err_low_map - norm_err_high_map

    return {
        "orig": orig_img_np,
        "norm_err_low_map": norm_err_low_map,
        "norm_err_high_map": norm_err_high_map,
        "norm_diff_map": norm_diff_map,
        "raw_score": raw_diff_map.mean(),
        "norm_score": norm_diff_map.mean(),
        "energy_info": f"E_low: {energy_low:.1f} | E_high: {energy_high:.1f}",
        "path": img_path
    }

def visualize_and_stat_patterns(real_dir, fake_dir, checkpoint_path, num_samples=50):
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

    print(f"📸 시각화 및 통계 수집 시작: 총 {num_pairs}쌍 (평가 반전 적용)")

    stats = {
        'real_raw': [], 'fake_raw': [],
        'real_norm': [], 'fake_norm': []
    }

    for i in range(num_pairs):
        real_data = get_normalized_error_maps(real_files[i], model, device, T_STEP, TARGET_LOW, TARGET_HIGH)
        fake_data = get_normalized_error_maps(fake_files[i], model, device, T_STEP, TARGET_LOW, TARGET_HIGH)
        if real_data is None or fake_data is None: continue

        stats['real_raw'].append(real_data['raw_score'])
        stats['fake_raw'].append(fake_data['raw_score'])
        stats['real_norm'].append(real_data['norm_score'])
        stats['fake_norm'].append(fake_data['norm_score'])

        all_maps = [real_data["norm_err_low_map"], real_data["norm_err_high_map"], fake_data["norm_err_low_map"], fake_data["norm_err_high_map"]]
        vmin, vmax = min(m.min() for m in all_maps), max(m.max() for m in all_maps)
        all_diffs = [real_data["norm_diff_map"], fake_data["norm_diff_map"]]
        diff_abs_max = max(max(abs(m.min()), abs(m.max())) for m in all_diffs)

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        rows_data = [("REAL", real_data), ("FAKE", fake_data)]

        for row_idx, (label, data) in enumerate(rows_data):
            axes[row_idx, 0].imshow(data["orig"])
            axes[row_idx, 0].set_ylabel(label, fontsize=15, fontweight='bold', color='steelblue' if row_idx==0 else 'tomato')
            axes[row_idx, 0].set_title(f"Original\n({data['energy_info']})", fontsize=10)
            axes[row_idx, 0].set_xticks([]); axes[row_idx, 0].set_yticks([])

            axes[row_idx, 1].imshow(data["norm_err_low_map"], cmap='hot', vmin=vmin, vmax=vmax)
            axes[row_idx, 1].set_title(f"Norm {TARGET_LOW[0]} Error" if row_idx==0 else "")
            axes[row_idx, 1].set_xticks([]); axes[row_idx, 1].set_yticks([])

            axes[row_idx, 2].imshow(data["norm_err_high_map"], cmap='hot', vmin=vmin, vmax=vmax)
            axes[row_idx, 2].set_title(f"Norm {TARGET_HIGH[0]} Error" if row_idx==0 else "")
            axes[row_idx, 2].set_xticks([]); axes[row_idx, 2].set_yticks([])

            # 🚨 [수정됨] 타이틀 반영 (Low - High)
            im3 = axes[row_idx, 3].imshow(data["norm_diff_map"], cmap='seismic', vmin=-diff_abs_max, vmax=diff_abs_max)
            axes[row_idx, 3].set_title("Normalized Diff Map (Low - High)" if row_idx==0 else "")
            
            score_text = f"Raw Score: {data['raw_score']:.4f}\nNorm Score: {data['norm_score']:.6f}"
            color = 'blue' if data['norm_score'] > 0 else 'red'
            axes[row_idx, 3].set_xlabel(score_text, fontweight='bold', color=color, fontsize=11)
            axes[row_idx, 3].set_xticks([]); axes[row_idx, 3].set_yticks([])

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        # 🚨 [수정됨] 컬러바 라벨 반영
        fig.colorbar(im3, cax=cbar_ax, label='Normalized Diff Value (Low - High)')

        plt.suptitle(f"Reversed Pattern | Pair #{i+1} (t={T_STEP})\nReal: {os.path.basename(real_data['path'])} vs Fake: {os.path.basename(fake_data['path'])}", fontsize=15, fontweight='bold')
        plt.subplots_adjust(right=0.9)
        plt.savefig(os.path.join(SAVE_DIR, f"norm_pattern_pair_{i+1:02d}.png"), dpi=150)
        plt.close(fig)

    if len(stats['real_norm']) > 0:
        # 🚨 [수정됨] 정확도(Accuracy) 기준 반전: Fake의 점수가 Real보다 높은 경우를 정답으로 간주
        raw_correct = sum(f > r for r, f in zip(stats['real_raw'], stats['fake_raw']))
        norm_correct = sum(f > r for r, f in zip(stats['real_norm'], stats['fake_norm']))
        
        raw_acc = (raw_correct / num_pairs) * 100
        norm_acc = (norm_correct / num_pairs) * 100

        print("\n" + "="*50)
        print("📊 [통계 요약 결과 - 평가 반전]")
        print(f"총 테스트 샘플: {num_pairs}쌍")
        print(f"Raw Score (Mean) -> Real: {np.mean(stats['real_raw']):.5f} | Fake: {np.mean(stats['fake_raw']):.5f}")
        print(f"Norm Score (Mean)-> Real: {np.mean(stats['real_norm']):.6f} | Fake: {np.mean(stats['fake_norm']):.6f}")
        print("-" * 50)
        # 🚨 [수정됨] 콘솔 출력 텍스트 반영
        print(f"🎯 Raw 가설 일치율 (Fake > Real): {raw_acc:.1f}% ({raw_correct}/{num_pairs})")
        print(f"🎯 Norm 가설 일치율 (Fake > Real): {norm_acc:.1f}% ({norm_correct}/{num_pairs})")
        print("="*50)

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        ax[0].bar(['Real', 'Fake'], [np.mean(stats['real_raw']), np.mean(stats['fake_raw'])], color=['steelblue', 'tomato'])
        ax[0].set_title(f"Reversed Raw Score Mean\nAccuracy: {raw_acc:.1f}%")
        ax[0].set_ylabel('Mean Raw Score (Low - High)')
        
        ax[1].bar(['Real', 'Fake'], [np.mean(stats['real_norm']), np.mean(stats['fake_norm'])], color=['skyblue', 'lightsalmon'])
        ax[1].set_title(f"Reversed Normalized Score Mean\nAccuracy: {norm_acc:.1f}%")
        ax[1].set_ylabel('Mean Norm Score (Low - High)')

        plt.suptitle(f"Reversed Statistics for 50 Pairs (nano_banana)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "summary_statistics_reversed.png"), dpi=150)
        plt.close(fig)

    print(f"\n✅ 통계 요약 완료! '{SAVE_DIR}/summary_statistics_reversed.png'를 확인하세요.")

# --- 실행부 ---
MY_CHECKPOINT = "/data1/checkpoints/HYRE/past_mymodel5_bin_ep66_acc0.8856.pth"

visualize_and_stat_patterns(
    real_dir="/data1/DeepFake/REAL/DIV2K",
    fake_dir="/data1/DeepFake/FAKE/nano_banana",
    checkpoint_path=MY_CHECKPOINT,
    num_samples=NUM_SAMPLES,
)