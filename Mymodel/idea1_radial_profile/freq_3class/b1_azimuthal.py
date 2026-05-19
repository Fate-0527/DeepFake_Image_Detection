"""
B-1: 방위각(Azimuthal) 프로파일 분석
- FFT 이미지를 각도별 섹터로 나눠 에너지 합산
- Real vs Old Fake vs New Fake의 각도 에너지 분포 차이 시각화
- 주요 Spike 각도 기반 SVM 분류
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

SEED       = 42
N_ANGLES   = 36        # 10도 단위 (36 섹터)
NUM_SAMPLES = 500      # 클래스당 이미지 수 (빠른 실행용)
IMG_SIZE   = Config.IMG_SIZE
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'freq_analysis_outputs', 'figures')
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

# ─── 캐시 경로 ────────────────────────────────────────────────
CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'freq_analysis_outputs', 'azimuthal_features.pkl')


def get_azimuthal_profile(fft_mag, n_angles=N_ANGLES):
    """
    2D FFT magnitude 맵의 방위각(각도)별 평균 에너지 계산.
    중심(DC)을 기준으로 0~360도를 n_angles 개 섹터로 분할.
    """
    H, W = fft_mag.shape
    cy, cx = H // 2, W // 2
    y_idx, x_idx = np.indices((H, W))
    dy = y_idx - cy
    dx = x_idx - cx
    angle = np.degrees(np.arctan2(dy, dx)) % 360  # 0~360도
    angle_bin = (angle / (360 / n_angles)).astype(int)
    angle_bin = np.clip(angle_bin, 0, n_angles - 1)

    profile = np.zeros(n_angles)
    for k in range(n_angles):
        mask = (angle_bin == k)
        if mask.sum() > 0:
            profile[k] = fft_mag[mask].mean()
    return profile


def compute_azimuthal_from_image(img_path, transform):
    """이미지 파일 경로 → 방위각 프로파일 벡터."""
    img = Image.open(img_path).convert('RGB')
    t = transform(img)
    gray = t.mean(dim=0).numpy()
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log(np.abs(f_shift) + 1e-8)
    return get_azimuthal_profile(mag)


def collect_profiles(dirs_list, label, n_per_class, transform, desc):
    files = []
    for d in dirs_list:
        fs = sorted([f for f in glob.glob(os.path.join(d, "*.*"))
                     if f.lower().endswith(VALID_EXTS)])
        files.extend(fs)
    files = files[:n_per_class]

    profiles, labels = [], []
    for fp in tqdm(files, desc=f"[{desc}]", ncols=80):
        try:
            p = compute_azimuthal_from_image(fp, transform)
            profiles.append(p)
            labels.append(label)
        except Exception:
            pass
    return np.array(profiles), np.array(labels)


def run():
    print("=" * 60)
    print("  B-1: 방위각(Azimuthal) 프로파일 분석")
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
        X, y = cache['X'], cache['y']
    else:
        n_old = NUM_SAMPLES // len(OLD_FAKE_MODELS)
        n_new = NUM_SAMPLES // len(NEW_FAKE_MODELS)

        # Real
        Xr, yr = collect_profiles(REAL_DIRS, LABEL_REAL, NUM_SAMPLES, transform, "REAL")
        # Old Fake
        Xo_list, yo_list = [], []
        for m in OLD_FAKE_MODELS:
            xo, yo = collect_profiles([FAKE_DIRS[m]], LABEL_OLD_FAKE, n_old, transform, f"OldFake/{m}")
            Xo_list.append(xo); yo_list.append(yo)
        Xo, yo = np.concatenate(Xo_list), np.concatenate(yo_list)
        # New Fake
        Xn_list, yn_list = [], []
        for m in NEW_FAKE_MODELS:
            xn, yn = collect_profiles([FAKE_DIRS[m]], LABEL_NEW_FAKE, n_new, transform, f"NewFake/{m}")
            Xn_list.append(xn); yn_list.append(yn)
        Xn, yn = np.concatenate(Xn_list), np.concatenate(yn_list)

        X = np.concatenate([Xr, Xo, Xn])
        y = np.concatenate([yr, yo, yn])

        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump({'X': X, 'y': y}, f)
        print(f"✅ 캐시 저장 완료: {CACHE_PATH}")

    print(f"  Total samples: {len(y)}  (Real={( y==0).sum()}, OldFake={( y==1).sum()}, NewFake={( y==2).sum()})")

    # ── 시각화 1: 클래스별 방위각 프로파일 (평균 ± std) ─────
    angles = np.linspace(0, 360, N_ANGLES, endpoint=False)
    names  = {0: 'REAL', 1: 'Old Fake', 2: 'New Fake'}
    colors = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}

    fig, ax = plt.subplots(figsize=(12, 5))
    for lbl in [0, 1, 2]:
        r = X[y == lbl]
        mu  = r.mean(axis=0)
        std = r.std(axis=0)
        ax.plot(angles, mu, label=names[lbl], color=colors[lbl], lw=2)
        ax.fill_between(angles, mu - std, mu + std, color=colors[lbl], alpha=0.15)
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Mean Log Magnitude', fontsize=12)
    ax.set_title('B-1: Azimuthal Profile (각도별 주파수 에너지, mean ± std)', fontsize=13, fontweight='bold')
    ax.set_xticks(np.arange(0, 361, 45))
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, 'b1_azimuthal_curves.png')
    plt.savefig(path1, dpi=150); plt.close()
    print(f"  📊 방위각 곡선 저장: {path1}")

    # ── 시각화 2: Spike 탐지 (클래스별 분산이 높은 각도 강조) ─
    fig, ax = plt.subplots(figsize=(12, 5))
    for lbl in [0, 1, 2]:
        r = X[y == lbl]
        variance = r.var(axis=0)
        ax.plot(angles, variance, label=f'{names[lbl]} Variance', color=colors[lbl], lw=2)
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Variance of Log Magnitude', fontsize=12)
    ax.set_title('B-1: 각도별 분산 (클래스 내부 얼마나 흩어져 있나)', fontsize=13, fontweight='bold')
    ax.set_xticks(np.arange(0, 361, 45))
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, 'b1_azimuthal_variance.png')
    plt.savefig(path2, dpi=150); plt.close()
    print(f"  📊 분산 곡선 저장: {path2}")

    # ── 시각화 3: Real 대비 Residual (방위각 기준) ───────────
    real_mean = X[y == 0].mean(axis=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    for lbl in [1, 2]:
        r = X[y == lbl]
        residual_mu  = (r - real_mean).mean(axis=0)
        residual_std = (r - real_mean).std(axis=0)
        ax.plot(angles, residual_mu, label=names[lbl], color=colors[lbl], lw=2)
        ax.fill_between(angles, residual_mu - residual_std, residual_mu + residual_std,
                        color=colors[lbl], alpha=0.15)
    ax.axhline(0, color='black', ls='-', lw=0.8)
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Residual (Fake - Real mean)', fontsize=12)
    ax.set_title('B-1: 방위각 Residual (Real 평균 대비 Fake 차이)', fontsize=13, fontweight='bold')
    ax.set_xticks(np.arange(0, 361, 45))
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, 'b1_azimuthal_residual.png')
    plt.savefig(path3, dpi=150); plt.close()
    print(f"  📊 방위각 Residual 저장: {path3}")

    # ── 분류: SVM ─────────────────────────────────────────
    print("\n  [SVM 분류]")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
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
    ax.set_title('B-1 Azimuthal SVM Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path4 = os.path.join(OUTPUT_DIR, 'b1_svm_confusion.png')
    plt.savefig(path4, dpi=150); plt.close()
    print(f"  📊 Confusion Matrix 저장: {path4}")

    print("\n✅ B-1 완료!\n")


if __name__ == "__main__":
    run()
