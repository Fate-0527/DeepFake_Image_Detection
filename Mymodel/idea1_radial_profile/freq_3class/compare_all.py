"""
Phase 6: A-1 ~ A-5 종합 성능 비교
- 각 방법을 순서대로 실행하고 결과를 집계
- 최종 성능 비교표 + PCA 시각화
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import PCA
import xgboost as xgb

from feature_extractor import extract_and_save, LABEL_REAL
from scipy.stats import skew, kurtosis

SEED       = 42
HF_LOW     = 250
BANDS      = [(0, 50), (50, 150), (150, 250), (250, None)]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'freq_3class_outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── 각 방법의 피처 추출 함수 모음 ──────────────────────────────

def feat_a1(profiles):
    hf  = profiles[:, HF_LOW:].mean(axis=1)
    mid = profiles[:, 50:HF_LOW].mean(axis=1)
    return (hf / (mid + 1e-8)).reshape(-1, 1)

def feat_a2(profiles):
    L = profiles.shape[1]
    return np.concatenate(
        [profiles[:, lo:hi if hi else L].mean(axis=1, keepdims=True)
         for lo, hi in BANDS], axis=1)

def feat_a3_stat(profiles, real_mean):
    res = profiles - real_mean[np.newaxis, :]
    hf  = res[:, HF_LOW:]
    return np.stack([hf.mean(1), hf.std(1),
                     np.array([skew(r) for r in hf]),
                     np.array([kurtosis(r) for r in hf])], axis=1)

def feat_a4(profiles):
    L     = profiles.shape[1]
    r_arr = np.arange(HF_LOW, L)
    log_r = np.log(r_arr + 1)
    feats = []
    for p in profiles:
        lm = p[HF_LOW:]
        if len(lm) < 5:
            feats.append([0., 0., 0.])
            continue
        c      = np.polyfit(log_r, lm, 1)
        y_fit  = np.polyval(c, log_r)
        ss_res = ((lm - y_fit)**2).sum()
        ss_tot = ((lm - lm.mean())**2).sum()
        feats.append([c[0], c[1], 1 - ss_res/(ss_tot+1e-8)])
    return np.array(feats)

def feat_a5(profiles_orig, profiles_err):
    o = feat_a2(profiles_orig)
    e = feat_a2(profiles_err)
    return np.concatenate([o, e], axis=1)


def evaluate_method(X_tr, X_te, y_tr, y_te, method='svm', tag=''):
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_tr)
    Xte_s  = scaler.transform(X_te)
    if method == 'svm':
        clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
    else:  # xgb
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                 use_label_encoder=False, eval_metric='mlogloss',
                                 random_state=SEED, n_jobs=-1)
    clf.fit(Xtr_s, y_tr)
    preds = clf.predict(Xte_s)
    acc   = accuracy_score(y_te, preds)
    f1    = f1_score(y_te, preds, average='macro')
    print(f"  {tag:<30s} Acc={acc:.4f}  Macro F1={f1:.4f}")
    return acc, f1


def plot_summary_table(results, fname):
    """성능 비교 막대그래프."""
    methods = list(results.keys())
    accs    = [v[0] for v in results.values()]
    f1s     = [v[1] for v in results.values()]
    x       = np.arange(len(methods))
    w       = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - w/2, accs, w, label='Accuracy', color='#1565C0', alpha=0.85)
    b2 = ax.bar(x + w/2, f1s,  w, label='Macro F1',  color='#2E7D32', alpha=0.85)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004,
                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9, rotation=15, ha='right')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Frequency Domain 3-Class Classification — Method Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150); plt.close()
    print(f"\n  📊 종합 비교 그래프 저장: {path}")


def plot_pca_grid(feat_dict, y, fname):
    """각 방법의 PCA 2D 산점도를 한 그림에."""
    n     = len(feat_dict)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    names  = {0: 'REAL', 1: 'Old Fake', 2: 'New Fake'}
    colors = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = axes.ravel()
    for i, (title, X) in enumerate(feat_dict.items()):
        sc    = StandardScaler(); Xs = sc.fit_transform(X)
        
        # 피처가 1개인 경우 (A-1 HF Ratio 등) PCA 2차원 축소가 불가능하므로 예외 처리
        if Xs.shape[1] < 2:
            X2d = np.zeros((Xs.shape[0], 2))
            X2d[:, 0] = Xs[:, 0]
            var_ratio_1, var_ratio_2 = 100.0, 0.0
        else:
            pca   = PCA(n_components=2, random_state=SEED)
            X2d   = pca.fit_transform(Xs)
            var_ratio_1 = pca.explained_variance_ratio_[0] * 100
            var_ratio_2 = pca.explained_variance_ratio_[1] * 100
            
        ax    = axes[i]
        for lbl in [0, 1, 2]:
            m = (y == lbl)
            ax.scatter(X2d[m, 0], X2d[m, 1], label=names[lbl],
                       color=colors[lbl], alpha=0.3, s=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(f'PC1 ({var_ratio_1:.1f}%)', fontsize=9)
        ax.set_ylabel(f'PC2 ({var_ratio_2:.1f}%)', fontsize=9)
        ax.legend(fontsize=8, markerscale=2); ax.grid(True, alpha=0.2)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('PCA 2D — 각 방법의 피처 공간', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=120); plt.close()
    print(f"  📊 PCA 그리드 저장: {path}")


def run():
    print("=" * 60)
    print("  Phase 6: 종합 비교 (A-1 ~ A-5)")
    print("=" * 60)

    data  = extract_and_save()
    Xo    = data['profiles_orig']
    Xe    = data['profiles_err']
    y     = data['labels']

    # Real 평균 (전체 Real 기준; 실제론 train 분할 후 계산해야 하나 탐색 목적)
    real_mean = Xo[y == LABEL_REAL].mean(axis=0)

    # ── 분할 ─────────────────────────────────────────────
    idx    = np.arange(len(y))
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    y_tr, y_te = y[i_tr], y[i_te]

    # Real 평균은 Train 셋 Real만으로 재계산 (엄밀하게)
    real_mean_tr = Xo[i_tr][y_tr == LABEL_REAL].mean(axis=0)

    # ── 피처 딕셔너리 ──────────────────────────────────────
    feat_sets = {
        'A-1 HF Ratio':          feat_a1(Xo),
        'A-2 4-band':             feat_a2(Xo),
        'A-3 Residual Stat':      feat_a3_stat(Xo, real_mean_tr),
        'A-4 Slope':              feat_a4(Xo),
        'A-5 Orig+Err 8-band':   feat_a5(Xo, Xe),
    }

    results = {}
    print("\n  [SVM 평가]")
    for name, X_all in feat_sets.items():
        X_tr_f, X_te_f = X_all[i_tr], X_all[i_te]
        acc, f1 = evaluate_method(X_tr_f, X_te_f, y_tr, y_te, 'svm', name)
        results[name + ' (SVM)'] = (acc, f1)

    print("\n  [XGBoost 평가]")
    for name, X_all in feat_sets.items():
        if name == 'A-1 HF Ratio':
            continue  # 1차원이라 XGB 의미 없음
        X_tr_f, X_te_f = X_all[i_tr], X_all[i_te]
        acc, f1 = evaluate_method(X_tr_f, X_te_f, y_tr, y_te, 'xgb', name)
        results[name + ' (XGB)'] = (acc, f1)

    # ── 최종 결과 테이블 ───────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  {'Method':<35s} {'Accuracy':>9s}  {'Macro F1':>9s}")
    print("  " + "-" * 60)
    for name, (acc, f1) in sorted(results.items(), key=lambda x: -x[1][1]):
        print(f"  {name:<35s} {acc:>9.4f}  {f1:>9.4f}")
    print("=" * 65)

    # ── 시각화 ────────────────────────────────────────────
    plot_summary_table(results, 'compare_all_methods.png')
    plot_pca_grid(feat_sets, y, 'compare_pca_grid.png')

    print("\n✅ 종합 비교 완료!")
    print(f"   결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
