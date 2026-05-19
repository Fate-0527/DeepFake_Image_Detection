"""
A-4: Spectral Slope 기반 3진 분류
- 고주파 구간(r > HF_LOW)에서 log-log 선형 피팅 → 기울기(slope)
- 분류: slope 단일 스칼라 Threshold + SVM(slope, intercept, R²)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

from feature_extractor import extract_and_save

HF_LOW     = 250
SEED       = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'freq_3class_outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_slope_features(profiles, hf_low=HF_LOW):
    """각 프로파일 고주파 구간 log-log 선형 피팅 → (slope, intercept, R²) 반환."""
    L     = profiles.shape[1]
    r_arr = np.arange(hf_low, L)
    log_r = np.log(r_arr + 1)
    feats = []
    for profile in profiles:
        log_mag = profile[hf_low:]
        if len(log_mag) < 5:
            feats.append([0.0, 0.0, 0.0])
            continue
        # 1차 선형 피팅
        p      = np.polyfit(log_r, log_mag, 1)
        slope, intercept = p[0], p[1]
        # R² 계산
        y_pred = np.polyval(p, log_r)
        ss_res = np.sum((log_mag - y_pred) ** 2)
        ss_tot = np.sum((log_mag - log_mag.mean()) ** 2)
        r2     = 1 - (ss_res / (ss_tot + 1e-8))
        feats.append([slope, intercept, r2])
    return np.array(feats)   # [N, 3]


def plot_confusion(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['REAL', 'OldFake', 'NewFake']
                           ).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 Confusion Matrix 저장: {path}")


def plot_slope_dist(slopes, y, fname):
    """클래스별 slope 박스플롯 + 바이올린플롯."""
    names    = ['REAL', 'Old Fake', 'New Fake']
    colors   = ['#2196F3', '#F44336', '#4CAF50']
    data_per = [slopes[y == lbl] for lbl in [0, 1, 2]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 박스플롯
    bp = axes[0].boxplot(data_per, labels=names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    axes[0].set_ylabel('Spectral Slope', fontsize=12)
    axes[0].set_title('A-4: Slope 분포 (Boxplot)', fontsize=13, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)

    # 바이올린플롯
    parts = axes[1].violinplot(data_per, showmedians=True)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color); pc.set_alpha(0.6)
    axes[1].set_xticks([1, 2, 3]); axes[1].set_xticklabels(names)
    axes[1].set_ylabel('Spectral Slope', fontsize=12)
    axes[1].set_title('A-4: Slope 분포 (Violin)', fontsize=13, fontweight='bold')
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 Slope 분포 저장: {path}")


def plot_example_fits(profiles, y, n=3, fname='a4_example_fits.png'):
    """대표 이미지 3개 × 3클래스의 고주파 스펙트럼 + 피팅 직선."""
    names  = {0: 'REAL', 1: 'Old Fake', 2: 'New Fake'}
    colors = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}
    L      = profiles.shape[1]
    r_arr  = np.arange(HF_LOW, L)
    log_r  = np.log(r_arr + 1)

    fig, axes = plt.subplots(n, 3, figsize=(14, 3*n), sharey=False)
    for col, lbl in enumerate([0, 1, 2]):
        idxs = np.where(y == lbl)[0][:n]
        for row, idx in enumerate(idxs):
            log_mag = profiles[idx, HF_LOW:]
            p       = np.polyfit(log_r, log_mag, 1)
            axes[row, col].plot(log_r, log_mag, alpha=0.7, color=colors[lbl], lw=1.5)
            axes[row, col].plot(log_r, np.polyval(p, log_r), 'k--', lw=1.2,
                                label=f'slope={p[0]:.3f}')
            axes[row, col].legend(fontsize=8)
            if row == 0:
                axes[row, col].set_title(names[lbl], fontsize=12, fontweight='bold',
                                         color=colors[lbl])
            if col == 0:
                axes[row, col].set_ylabel('Log Magnitude', fontsize=9)
    plt.suptitle('A-4: Example Log-Log Spectral Fits (HF region)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 피팅 예시 저장: {path}")


def run():
    print("=" * 60)
    print("  A-4: Spectral Slope 3진 분류")
    print("=" * 60)

    data  = extract_and_save()
    X_raw = data['profiles_orig']
    y     = data['labels']

    print("\n  Slope 피처 계산 중...")
    feats = compute_slope_features(X_raw)   # [N, 3]: slope, intercept, R²
    slopes = feats[:, 0]
    print(f"  Slope 범위: [{slopes.min():.4f}, {slopes.max():.4f}]")

    # ── 클래스별 통계 출력 ─────────────────────────────────
    for lbl, name in [(0, 'REAL'), (1, 'OldFake'), (2, 'NewFake')]:
        s = slopes[y == lbl]
        print(f"  [{name}] slope: mean={s.mean():.4f}, std={s.std():.4f}, "
              f"min={s.min():.4f}, max={s.max():.4f}")

    # ── 시각화 ────────────────────────────────────────────
    plot_slope_dist(slopes, y, 'a4_slope_distribution.png')
    plot_example_fits(X_raw, y, n=3)

    # ── Train / Test 분할 ──────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        feats, y, test_size=0.2, random_state=SEED, stratify=y
    )
    slopes_tr = X_tr[:, 0]
    slopes_te = X_te[:, 0]

    # ══════════════════════════════════════════════════════
    # [1] Slope 단일 Threshold 분류
    # ══════════════════════════════════════════════════════
    print("\n  [slope 단일 threshold 탐색]")
    real_mask  = (y_tr == 0)
    mu_s    = slopes_tr[real_mask].mean()
    sigma_s = slopes_tr[real_mask].std()
    print(f"  Real slope: μ={mu_s:.4f}, σ={sigma_s:.4f}")

    best_k, best_f1 = 0.5, 0.0
    for k in np.arange(0.2, 2.1, 0.1):
        preds_k = np.full(len(slopes_tr), 0, dtype=int)
        # 주의: 기울기가 완만(less negative) → Old Fake
        preds_k[slopes_tr > mu_s + k * sigma_s] = 1
        preds_k[slopes_tr < mu_s - k * sigma_s] = 2
        f1 = f1_score(y_tr, preds_k, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_k = f1, k

    preds_thresh = np.full(len(slopes_te), 0, dtype=int)
    preds_thresh[slopes_te > mu_s + best_k * sigma_s] = 1
    preds_thresh[slopes_te < mu_s - best_k * sigma_s] = 2
    print(f"\n  최적 k={best_k:.1f} (Train F1={best_f1:.4f})")
    print("  [Threshold Test 결과]")
    print(classification_report(y_te, preds_thresh,
                                target_names=['REAL', 'OldFake', 'NewFake'], digits=4))
    plot_confusion(y_te, preds_thresh,
                   f'A-4 Slope Threshold (k={best_k:.1f})', 'a4_threshold_confusion.png')

    # ══════════════════════════════════════════════════════
    # [2] (slope, intercept, R²) 3차원 → SVM
    # ══════════════════════════════════════════════════════
    print("\n  [SVM (slope + intercept + R²)]")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    svm    = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
    svm.fit(X_tr_s, y_tr)
    preds_svm = svm.predict(X_te_s)
    print(classification_report(y_te, preds_svm,
                                target_names=['REAL', 'OldFake', 'NewFake'], digits=4))
    plot_confusion(y_te, preds_svm, 'A-4 SVM Confusion Matrix', 'a4_svm_confusion.png')

    print("\n✅ A-4 완료!\n")


if __name__ == "__main__":
    run()
