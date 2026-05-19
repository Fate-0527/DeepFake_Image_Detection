"""
B-4: 위상(Phase) 일관성 지표 분석
- FFT 위상 스펙트럼의 인접 픽셀 간 위상 차이(Phase Gradient) 분산 계산
- "위상이 얼마나 갑자기 튀는가" → Fake 이미지에서 불연속성이 더 크게 나타남
- 반경별 위상 일관성 곡선 + SVM 분류
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle
import glob
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm

from config import Config

SEED        = 42
N_BINS      = 50        # 반경 방향 Bin 수
NUM_SAMPLES = 500       # 클래스당 이미지 수
IMG_SIZE    = Config.IMG_SIZE
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'freq_analysis_outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_REAL     = 0
LABEL_OLD_FAKE = 1
LABEL_NEW_FAKE = 2

OLD_FAKE_MODELS = ["OpenJourney", "SD1_4", "SD1_5"]
NEW_FAKE_MODELS = ["flux_dev", "nano_banana", "std_3.5_large_turbo"]
FAKE_DIRS = {
    "OpenJourney":         "/data1/DeepFake/FAKE/OpenJourney",
    "SD1_4":               "/data1/DeepFake/FAKE/SD1_4",
    "SD1_5":               "/data1/DeepFake/FAKE/SD1_5",
    "flux_dev":            "/data1/DeepFake/FAKE/flux_dev",
    "nano_banana":         "/data1/DeepFake/FAKE/nano_banana",
    "std_3.5_large_turbo": "/data1/DeepFake/FAKE/std_3.5_large_turbo",
}
REAL_DIRS = [
    "/data1/DeepFake/REAL/DIV2K",
    "/data1/DeepFake/REAL/CLIC",
    "/data1/DeepFake/REAL/Flickr2K",
    "/data1/DeepFake/REAL/LSDIR",
    "/data1/DeepFake/REAL/RAISE",
    "/data1/DeepFake/REAL/UCID1338",
]
VALID_EXTS = ('.png', '.jpg', '.jpeg', '.webp')
CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'freq_analysis_outputs', 'phase_features.pkl')


def compute_phase_consistency_profile(img_gray, n_bins=N_BINS):
    """
    이미지(회색) → FFT 위상 맵 → 반경별 위상 Gradient 분산 프로파일.
    각 반경 Bin에서 인접 픽셀 위상 차이(|∇phase|)의 분산을 계산.
    반환: [n_bins] 벡터
    """
    H, W = img_gray.shape
    cy, cx = H // 2, W // 2

    f_shift = np.fft.fftshift(np.fft.fft2(img_gray))
    phase   = np.angle(f_shift)   # -π ~ π

    # 위상 Gradient (인접 픽셀 차이, 각도 wrapping 고려)
    dy = np.diff(phase, axis=0, prepend=phase[:1, :])
    dx = np.diff(phase, axis=1, prepend=phase[:, :1])
    # Wrapping: [-π, π] 범위로 조정
    dy = (dy + np.pi) % (2 * np.pi) - np.pi
    dx = (dx + np.pi) % (2 * np.pi) - np.pi
    grad_mag = np.sqrt(dy**2 + dx**2)   # [H, W]

    # 반경 맵
    y_idx, x_idx = np.indices((H, W))
    r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    max_r = np.sqrt(cx**2 + cy**2)
    bin_edges = np.linspace(0, max_r, n_bins + 1)

    profile = np.zeros(n_bins)
    for k in range(n_bins):
        mask = (r >= bin_edges[k]) & (r < bin_edges[k + 1])
        if mask.sum() > 0:
            profile[k] = grad_mag[mask].var()   # 분산 = 얼마나 불규칙하게 튀나
    return profile


def compute_phase_stats(img_gray):
    """
    전체 요약 통계 Feature [6개]:
    - 전체 Phase Gradient 분산
    - 고주파(외곽 50%) Phase Gradient 분산
    - 위상 엔트로피 (분포의 복잡도)
    - 위상 std
    - 위상 Gradient 평균
    - 위상 Gradient 최대/평균 비율
    """
    H, W = img_gray.shape
    cy, cx = H // 2, W // 2
    f_shift = np.fft.fftshift(np.fft.fft2(img_gray))
    phase   = np.angle(f_shift)
    dy = (np.diff(phase, axis=0, prepend=phase[:1, :]) + np.pi) % (2*np.pi) - np.pi
    dx = (np.diff(phase, axis=1, prepend=phase[:, :1]) + np.pi) % (2*np.pi) - np.pi
    grad_mag = np.sqrt(dy**2 + dx**2)

    y_idx, x_idx = np.indices((H, W))
    r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    max_r = r.max()
    hf_mask = r >= max_r * 0.5

    # 위상 히스토그램 엔트로피
    hist, _ = np.histogram(phase.ravel(), bins=64, range=(-np.pi, np.pi), density=True)
    hist    = hist + 1e-10
    entropy = -np.sum(hist * np.log(hist))

    return np.array([
        grad_mag.var(),
        grad_mag[hf_mask].var() if hf_mask.sum() > 0 else 0.0,
        entropy,
        phase.std(),
        grad_mag.mean(),
        grad_mag.max() / (grad_mag.mean() + 1e-10),
    ])


def compute_features_from_image(img_path, transform):
    img  = Image.open(img_path).convert('RGB')
    t    = transform(img)
    gray = t.mean(dim=0).numpy()
    profile = compute_phase_consistency_profile(gray)
    stats   = compute_phase_stats(gray)
    return profile, stats


def collect_all(dirs_list, label, n, transform, desc):
    files = []
    for d in dirs_list:
        files.extend(sorted([f for f in glob.glob(os.path.join(d, "*.*"))
                             if f.lower().endswith(VALID_EXTS)]))
    files = files[:n]
    profiles, stats_list, labels = [], [], []
    for fp in tqdm(files, desc=f"[{desc}]", ncols=80):
        try:
            p, s = compute_features_from_image(fp, transform)
            profiles.append(p); stats_list.append(s); labels.append(label)
        except Exception:
            pass
    return np.array(profiles), np.array(stats_list), np.array(labels)


def run():
    print("=" * 60)
    print("  B-4: 위상(Phase) 일관성 지표 분석")
    print("=" * 60)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # ── 캐시 확인 ─────────────────────────────────────────
    if os.path.exists(CACHE_PATH):
        print(f"✅ 캐시 로드: {CACHE_PATH}")
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        X_profile, X_stats, y = cache['profiles'], cache['stats'], cache['y']
    else:
        n_old = NUM_SAMPLES // len(OLD_FAKE_MODELS)
        n_new = NUM_SAMPLES // len(NEW_FAKE_MODELS)

        Pr, Sr, yr = collect_all(REAL_DIRS, LABEL_REAL, NUM_SAMPLES, transform, "REAL")
        Po_list, So_list, yo_list = [], [], []
        for m in OLD_FAKE_MODELS:
            p, s, lbl = collect_all([FAKE_DIRS[m]], LABEL_OLD_FAKE, n_old, transform, f"OldFake/{m}")
            Po_list.append(p); So_list.append(s); yo_list.append(lbl)
        Pn_list, Sn_list, yn_list = [], [], []
        for m in NEW_FAKE_MODELS:
            p, s, lbl = collect_all([FAKE_DIRS[m]], LABEL_NEW_FAKE, n_new, transform, f"NewFake/{m}")
            Pn_list.append(p); Sn_list.append(s); yn_list.append(lbl)

        X_profile = np.concatenate([Pr] + Po_list + Pn_list)
        X_stats   = np.concatenate([Sr] + So_list + Sn_list)
        y         = np.concatenate([yr] + yo_list + yn_list)

        with open(CACHE_PATH, 'wb') as f:
            pickle.dump({'profiles': X_profile, 'stats': X_stats, 'y': y}, f)
        print(f"✅ 캐시 저장 완료: {CACHE_PATH}")

    print(f"  Total: {len(y)}  (Real={( y==0).sum()}, OldFake={( y==1).sum()}, NewFake={( y==2).sum()})")

    names  = {0: 'REAL', 1: 'Old Fake', 2: 'New Fake'}
    colors = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}

    # ── 시각화 1: 반경별 위상 Gradient 분산 곡선 ──────────────
    bin_centers = np.linspace(0, 1.0, N_BINS)  # 정규화된 반경
    fig, ax = plt.subplots(figsize=(12, 5))
    for lbl in [0, 1, 2]:
        r   = X_profile[y == lbl]
        mu  = r.mean(axis=0)
        std = r.std(axis=0)
        ax.plot(bin_centers, mu, label=names[lbl], color=colors[lbl], lw=2)
        ax.fill_between(bin_centers, mu - std, mu + std, color=colors[lbl], alpha=0.15)
    ax.set_xlabel('Normalized Frequency Radius', fontsize=12)
    ax.set_ylabel('Phase Gradient Variance', fontsize=12)
    ax.set_title('B-4: 반경별 위상 Gradient 분산 (mean ± std)', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, 'b4_phase_gradient_profile.png')
    plt.savefig(path1, dpi=150); plt.close()
    print(f"  📊 위상 Gradient 프로파일 저장: {path1}")

    # ── 시각화 2: Real 대비 Residual ──────────────────────────
    real_mu = X_profile[y == 0].mean(axis=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    for lbl in [1, 2]:
        r = X_profile[y == lbl]
        res_mu  = (r - real_mu).mean(axis=0)
        res_std = (r - real_mu).std(axis=0)
        ax.plot(bin_centers, res_mu, label=names[lbl], color=colors[lbl], lw=2)
        ax.fill_between(bin_centers, res_mu - res_std, res_mu + res_std, color=colors[lbl], alpha=0.15)
    ax.axhline(0, color='black', ls='-', lw=0.8)
    ax.set_xlabel('Normalized Frequency Radius', fontsize=12)
    ax.set_ylabel('Residual Phase Gradient Variance', fontsize=12)
    ax.set_title('B-4: 위상 일관성 Residual (Real 대비)', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, 'b4_phase_residual.png')
    plt.savefig(path2, dpi=150); plt.close()
    print(f"  📊 위상 Residual 저장: {path2}")

    # ── 시각화 3: 요약 통계 박스플롯 ─────────────────────────
    stat_labels = ['Total Var', 'HF Var', 'Entropy', 'Phase Std', 'Grad Mean', 'Grad Max/Mean']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, (slbl, ax) in enumerate(zip(stat_labels, axes.flatten())):
        data_by_class = [X_stats[y == lbl, i] for lbl in [0, 1, 2]]
        bp = ax.boxplot(data_by_class, labels=['REAL', 'OldFake', 'NewFake'],
                        patch_artist=True, medianprops=dict(color='red', lw=2))
        for patch, color in zip(bp['boxes'], ['#2196F3', '#F44336', '#4CAF50']):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.set_title(slbl, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    plt.suptitle('B-4: 위상 일관성 통계 지표별 클래스 분포', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, 'b4_phase_stats_boxplot.png')
    plt.savefig(path3, dpi=150); plt.close()
    print(f"  📊 위상 통계 박스플롯 저장: {path3}")

    # ── 분류: SVM (profile + stats 결합) ──────────────────────
    # Profile 요약 통계 (mean, std, max) + stats
    prof_summary = np.hstack([
        X_profile.mean(axis=1, keepdims=True),
        X_profile.std(axis=1, keepdims=True),
        X_profile[:, N_BINS//2:].mean(axis=1, keepdims=True),   # 고주파 반쪽 평균
        X_stats
    ])

    print("\n  [SVM 분류 (Profile 요약 + Phase 통계)]")
    X_tr, X_te, y_tr, y_te = train_test_split(
        prof_summary, y, test_size=0.2, random_state=SEED, stratify=y)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
    svm.fit(X_tr_s, y_tr)
    preds = svm.predict(X_te_s)
    print(classification_report(y_te, preds, target_names=['REAL', 'OldFake', 'NewFake'], digits=4))

    cm = confusion_matrix(y_te, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['REAL', 'OldFake', 'NewFake']).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('B-4 Phase Consistency SVM Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path4 = os.path.join(OUTPUT_DIR, 'b4_svm_confusion.png')
    plt.savefig(path4, dpi=150); plt.close()
    print(f"  📊 Confusion Matrix 저장: {path4}")

    print("\n✅ B-4 완료!\n")


if __name__ == "__main__":
    run()
