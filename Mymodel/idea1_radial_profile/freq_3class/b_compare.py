"""
B-compare: 4가지 새 분석 방법 종합 비교
- B-1: Azimuthal (방위각) 프로파일
- B-2: HF Bin Variance (고주파 분산)
- B-3: Band Ratio (대역 비율)
- B-4: Phase Consistency (위상 일관성)
- 각 방법의 SVM 정확도를 한눈에 비교
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle

# 각 모듈에서 핵심 함수만 import
from b2_hf_variance import compute_bin_stats
from b3_band_ratio   import compute_band_ratios
from feature_extractor import extract_and_save

SEED       = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'freq_analysis_outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

AZIMUTHAL_CACHE = os.path.join(os.path.dirname(__file__), '..', 'freq_analysis_outputs', 'azimuthal_features.pkl')
PHASE_CACHE     = os.path.join(os.path.dirname(__file__), '..', 'freq_analysis_outputs', 'phase_features.pkl')


def svm_cv_score(X, y, n_splits=5):
    """Stratified K-Fold SVM 교차 검증."""
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    accs = []
    f1s  = []
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s  = sc.transform(X_te)
        clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
        clf.fit(X_tr_s, y_tr)
        preds = clf.predict(X_te_s)
        accs.append(accuracy_score(y_te, preds))
        f1s.append(f1_score(y_te, preds, average='macro'))
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


def run():
    print("=" * 60)
    print("  B-compare: 4가지 분석 방법 종합 비교")
    print("=" * 60)

    results = {}

    # ── B-2: HF Bin Variance ───────────────────────────────
    print("\n[B-2] HF Bin Variance ...")
    data  = extract_and_save()
    X_raw = data['profiles_orig']
    y     = data['labels']
    feat_b2 = compute_bin_stats(X_raw)
    acc, acc_std, f1, f1_std = svm_cv_score(feat_b2, y)
    results['B-2\nHF Variance'] = (acc, acc_std, f1, f1_std)
    print(f"  Acc={acc:.4f}±{acc_std:.4f}  F1={f1:.4f}±{f1_std:.4f}")

    # ── B-3: Band Ratio ────────────────────────────────────
    print("\n[B-3] Band Ratio ...")
    feat_b3 = compute_band_ratios(X_raw)
    acc, acc_std, f1, f1_std = svm_cv_score(feat_b3, y)
    results['B-3\nBand Ratio'] = (acc, acc_std, f1, f1_std)
    print(f"  Acc={acc:.4f}±{acc_std:.4f}  F1={f1:.4f}±{f1_std:.4f}")

    # ── B-1: Azimuthal ─────────────────────────────────────
    if os.path.exists(AZIMUTHAL_CACHE):
        print("\n[B-1] Azimuthal (캐시 로드) ...")
        with open(AZIMUTHAL_CACHE, 'rb') as f:
            c = pickle.load(f)
        X_az, y_az = c['X'], c['y']
        acc, acc_std, f1, f1_std = svm_cv_score(X_az, y_az)
        results['B-1\nAzimuthal'] = (acc, acc_std, f1, f1_std)
        print(f"  Acc={acc:.4f}±{acc_std:.4f}  F1={f1:.4f}±{f1_std:.4f}")
    else:
        print("\n[B-1] 캐시 없음 → b1_azimuthal.py 먼저 실행 필요")
        results['B-1\nAzimuthal'] = (0.0, 0.0, 0.0, 0.0)

    # ── B-4: Phase Consistency ─────────────────────────────
    if os.path.exists(PHASE_CACHE):
        print("\n[B-4] Phase Consistency (캐시 로드) ...")
        with open(PHASE_CACHE, 'rb') as f:
            c = pickle.load(f)
        X_ph, X_stats, y_ph = c['profiles'], c['stats'], c['y']
        N_BINS = X_ph.shape[1]
        feat_b4 = np.hstack([
            X_ph.mean(axis=1, keepdims=True),
            X_ph.std(axis=1, keepdims=True),
            X_ph[:, N_BINS//2:].mean(axis=1, keepdims=True),
            X_stats
        ])
        acc, acc_std, f1, f1_std = svm_cv_score(feat_b4, y_ph)
        results['B-4\nPhase'] = (acc, acc_std, f1, f1_std)
        print(f"  Acc={acc:.4f}±{acc_std:.4f}  F1={f1:.4f}±{f1_std:.4f}")
    else:
        print("\n[B-4] 캐시 없음 → b4_phase_consistency.py 먼저 실행 필요")
        results['B-4\nPhase'] = (0.0, 0.0, 0.0, 0.0)

    # ── 기존 A-3 Residual과도 비교 ─────────────────────────────
    from sklearn.model_selection import train_test_split
    from scipy.stats import skew, kurtosis
    print("\n[A-3 Residual (기존)] ...")
    real_mean = X_raw[y == 0].mean(axis=0)
    residuals = X_raw - real_mean
    hf = residuals[:, 250:]
    feat_a3 = np.stack([
        hf.mean(axis=1),
        hf.std(axis=1),
        np.array([skew(row) for row in hf]),
        np.array([kurtosis(row) for row in hf]),
    ], axis=1)
    acc, acc_std, f1, f1_std = svm_cv_score(feat_a3, y)
    results['A-3\nResidual'] = (acc, acc_std, f1, f1_std)
    print(f"  Acc={acc:.4f}±{acc_std:.4f}  F1={f1:.4f}±{f1_std:.4f}")

    # ── 시각화: 메서드별 Accuracy & F1 막대 그래프 ─────────────
    methods = list(results.keys())
    accs    = [results[m][0] for m in methods]
    acc_errs = [results[m][1] for m in methods]
    f1s     = [results[m][2] for m in methods]
    f1_errs  = [results[m][3] for m in methods]

    x = np.arange(len(methods))
    w = 0.35
    palette = ['#4CAF50' if 'B-' in m else '#FF9800' for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = ax1.bar(x - w/2, accs, w, yerr=acc_errs, capsize=5,
                    color=palette, alpha=0.8, label='Accuracy', edgecolor='white')
    ax1.set_xticks(x); ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('5-Fold CV Accuracy', fontsize=12)
    ax1.set_title('방법별 분류 정확도 비교', fontsize=13, fontweight='bold')
    ax1.axhline(0.33, color='gray', ls='--', lw=1, label='Random (33%)')
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()

    bars2 = ax2.bar(x - w/2, f1s, w, yerr=f1_errs, capsize=5,
                    color=palette, alpha=0.8, edgecolor='white')
    ax2.set_xticks(x); ax2.set_xticklabels(methods, fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('Macro F1 Score', fontsize=12)
    ax2.set_title('방법별 Macro F1 비교', fontsize=13, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    new_patch  = mpatches.Patch(color='#4CAF50', alpha=0.8, label='새 방법 (B 시리즈)')
    base_patch = mpatches.Patch(color='#FF9800', alpha=0.8, label='기존 방법 (A-3)')
    fig.legend(handles=[new_patch, base_patch], loc='lower center', ncol=2, fontsize=11, framealpha=0.9)

    plt.suptitle('B-1~B-4 vs A-3: 주파수 분석 방법 종합 비교\n(SVM, 5-Fold CV, mean ± std)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'b_compare_all.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n  📊 종합 비교 저장: {path}")

    # ── 최종 요약 출력 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📋 최종 성능 요약 (5-Fold CV)")
    print("=" * 60)
    print(f"  {'방법':<20} {'Accuracy':>12} {'Macro F1':>12}")
    print("-" * 48)
    for m in methods:
        acc, acc_std, f1, f1_std = results[m]
        tag = m.replace('\n', ' ')
        print(f"  {tag:<20} {acc:.4f}±{acc_std:.4f}  {f1:.4f}±{f1_std:.4f}")
    best = max(methods, key=lambda m: results[m][0])
    print(f"\n  🏆 Best: {best.replace(chr(10), ' ')} (Acc={results[best][0]:.4f})")
    print("=" * 60)
    print("\n✅ B-compare 완료!\n")


if __name__ == "__main__":
    run()
