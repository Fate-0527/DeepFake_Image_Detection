"""
A-1: 고주파 에너지 비율 (HF Energy Ratio) 기반 3진 분류
- 피처: HF_ratio = mean(profile[250:]) / mean(profile[50:250])
- 분류: Real 분포 μ±k*σ 기반 threshold
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy import stats

from feature_extractor import extract_and_save

# ─── 하이퍼파라미터 ────────────────────────────────────────
HF_LOW  = 250    # 고주파 시작 반경
MF_LOW  = 50     # 중주파 시작 반경
K_RANGE = np.arange(0.3, 2.1, 0.1)   # threshold k 탐색 범위
SEED    = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'freq_3class_outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_hf_ratio(profiles, hf_low=HF_LOW, mf_low=MF_LOW):
    """각 프로파일에 대해 HF_ratio 계산."""
    hf  = profiles[:, hf_low:].mean(axis=1)
    mid = profiles[:, mf_low:hf_low].mean(axis=1)
    return hf / (mid + 1e-8)


def threshold_classify(ratio, mu_real, sigma_real, k):
    """
    ratio > mu + k*sigma → Old Fake (1)
    ratio < mu - k*sigma → New Fake (2)
    else                 → Real     (0)
    """
    preds = np.full(len(ratio), 0, dtype=int)
    preds[ratio > mu_real + k * sigma_real] = 1
    preds[ratio < mu_real - k * sigma_real] = 2
    return preds


def find_best_k(ratio_train, y_train, mu_real, sigma_real):
    """Train 셋에서 macro F1이 최대인 k 탐색."""
    from sklearn.metrics import f1_score
    best_k, best_f1 = 0.5, 0.0
    for k in K_RANGE:
        preds = threshold_classify(ratio_train, mu_real, sigma_real, k)
        f1 = f1_score(y_train, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_k = f1, k
    return best_k, best_f1


def plot_histogram(ratio, labels, best_k, mu_real, sigma_real):
    """3클래스별 HF_ratio 히스토그램."""
    fig, ax = plt.subplots(figsize=(10, 5))
    names  = {0: 'REAL', 1: 'Old Fake', 2: 'New Fake'}
    colors = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}
    for lbl in [0, 1, 2]:
        ax.hist(ratio[labels == lbl], bins=60, alpha=0.55,
                label=names[lbl], color=colors[lbl], density=True)

    # Threshold 선 표시
    ax.axvline(mu_real + best_k * sigma_real, color='red',   ls='--', lw=1.5,
               label=f'Old Fake threshold (μ+{best_k:.1f}σ)')
    ax.axvline(mu_real - best_k * sigma_real, color='green', ls='--', lw=1.5,
               label=f'New Fake threshold (μ-{best_k:.1f}σ)')
    ax.axvline(mu_real, color='blue', ls='-', lw=1.0, label=f'Real mean')

    ax.set_xlabel('HF Energy Ratio', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('A-1: HF Energy Ratio Distribution per Class', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'a1_hf_ratio_histogram.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 히스토그램 저장: {path}")


def plot_confusion(y_true, y_pred, title, fname):
    """Confusion Matrix 시각화."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=['REAL', 'OldFake', 'NewFake'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Confusion Matrix 저장: {path}")


def run():
    print("=" * 60)
    print("  A-1: HF Energy Ratio 3진 분류")
    print("=" * 60)

    # ── 데이터 로드 ─────────────────────────────────────────
    data = extract_and_save()
    X    = data['profiles_orig']   # [N, L]
    y    = data['labels']          # [N]

    # ── 피처 계산 ─────────────────────────────────────────
    ratio = compute_hf_ratio(X)   # [N]

    # ── Train / Test 분할 ──────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        ratio.reshape(-1, 1), y,
        test_size=0.2, random_state=SEED, stratify=y
    )
    ratio_tr = X_tr.ravel()
    ratio_te = X_te.ravel()

    # ── Real 분포 추정 (Train 셋) ───────────────────────────
    real_mask  = (y_tr == 0)
    mu_real    = ratio_tr[real_mask].mean()
    sigma_real = ratio_tr[real_mask].std()
    print(f"\n  Real HF_ratio: μ={mu_real:.4f}, σ={sigma_real:.4f}")

    # ── 최적 k 탐색 ────────────────────────────────────────
    best_k, best_f1_train = find_best_k(ratio_tr, y_tr, mu_real, sigma_real)
    print(f"  최적 k={best_k:.1f}  (Train macro F1={best_f1_train:.4f})")

    # ── 테스트 평가 ────────────────────────────────────────
    preds_te = threshold_classify(ratio_te, mu_real, sigma_real, best_k)
    print("\n  [Test 결과]")
    print(classification_report(y_te, preds_te,
                                target_names=['REAL', 'OldFake', 'NewFake'],
                                digits=4))

    # ── 시각화 ─────────────────────────────────────────────
    plot_histogram(ratio, y, best_k, mu_real, sigma_real)
    plot_confusion(y_te, preds_te,
                   f'A-1 Confusion Matrix (k={best_k:.1f})',
                   'a1_confusion_matrix.png')

    # ── k vs F1 곡선 ──────────────────────────────────────
    from sklearn.metrics import f1_score
    f1_list = []
    for k in K_RANGE:
        preds_k = threshold_classify(ratio_te, mu_real, sigma_real, k)
        f1_list.append(f1_score(y_te, preds_k, average='macro', zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(K_RANGE, f1_list, marker='o', color='steelblue')
    ax.axvline(best_k, color='red', ls='--', label=f'Best k={best_k:.1f}')
    ax.set_xlabel('k', fontsize=12); ax.set_ylabel('Macro F1', fontsize=12)
    ax.set_title('A-1: Threshold k vs Macro F1 (Test)', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'a1_k_vs_f1.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 k vs F1 곡선 저장: {path}")

    print("\n✅ A-1 완료!\n")


if __name__ == "__main__":
    run()
