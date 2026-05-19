"""
A-5: FIRE 오차맵 FFT + 원본 FFT 병합 기반 3진 분류
- 피처: orig 4-band + err 4-band = 8차원
- 분류: SVM, XGBoost
- A-2(원본만) vs A-5(병합) 성능 비교
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import xgboost as xgb

from feature_extractor import extract_and_save

BANDS      = [(0, 50), (50, 150), (150, 250), (250, None)]
BAND_NAMES = ['Low', 'Mid', 'Mid-High', 'High']
SEED       = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'freq_3class_outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_multiband(profiles):
    """[N, L] → [N, 4] 멀티밴드 에너지."""
    L     = profiles.shape[1]
    feats = []
    for lo, hi in BANDS:
        hi_ = hi if hi is not None else L
        feats.append(profiles[:, lo:hi_].mean(axis=1, keepdims=True))
    return np.concatenate(feats, axis=1)


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


def plot_err_radial_curves(profiles_err, y, fname):
    """FIRE 오차맵 radial profile 클래스별 평균 곡선."""
    names  = {0: 'REAL', 1: 'Old Fake', 2: 'New Fake'}
    colors = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}
    x = np.arange(profiles_err.shape[1])
    fig, ax = plt.subplots(figsize=(12, 5))
    for lbl in [0, 1, 2]:
        r = profiles_err[y == lbl]
        mu  = r.mean(axis=0)
        std = r.std(axis=0)
        ax.plot(x, mu, label=names[lbl], color=colors[lbl], lw=2)
        ax.fill_between(x, mu - std, mu + std, color=colors[lbl], alpha=0.15)
    ax.axvline(250, color='gray', ls='--', lw=1.2, label='r=250')
    ax.set_xlabel('Frequency Radius (r)', fontsize=12)
    ax.set_ylabel('Log Magnitude (Error Map)', fontsize=12)
    ax.set_title('A-5: FIRE 오차맵 Radial Profile (클래스별)', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 오차맵 프로파일 저장: {path}")


def plot_performance_comparison(results, fname):
    """A-2 vs A-5 성능 비교 막대그래프."""
    methods = list(results.keys())
    accs    = [v['accuracy'] for v in results.values()]
    f1s     = [v['macro_f1'] for v in results.values()]
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='#1565C0', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1s,  width, label='Macro F1',  color='#388E3C', alpha=0.8)
    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('A-2 (orig only) vs A-5 (orig + FIRE err) 성능 비교', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 성능 비교 그래프 저장: {path}")


def run():
    print("=" * 60)
    print("  A-5: FIRE 오차맵 병합 3진 분류")
    print("=" * 60)

    data         = extract_and_save()
    X_orig       = data['profiles_orig']  # [N, L]
    X_err        = data['profiles_err']   # [N, L]
    y            = data['labels']

    # ── 피처 구성 ─────────────────────────────────────────
    feat_orig = extract_multiband(X_orig)   # [N, 4]
    feat_err  = extract_multiband(X_err)    # [N, 4]
    feat_a2   = feat_orig                   # A-2용 (원본만)
    feat_a5   = np.concatenate([feat_orig, feat_err], axis=1)  # A-5용 (8차원)
    print(f"  A-2 피처: {feat_a2.shape}  |  A-5 피처: {feat_a5.shape}")

    # ── 오차맵 프로파일 시각화 ─────────────────────────────
    plot_err_radial_curves(X_err, y, 'a5_err_radial_curves.png')

    # ── Train / Test 분할 (A-2, A-5 동일 시드) ────────────
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)

    results = {}

    for tag, feat_X in [('A-2 (4d, SVM)', feat_a2),
                        ('A-5 (8d, SVM)', feat_a5)]:
        X_tr = feat_X[idx_tr]; X_te = feat_X[idx_te]
        y_tr = y[idx_tr];      y_te = y[idx_te]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        svm    = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
        svm.fit(X_tr_s, y_tr)
        preds  = svm.predict(X_te_s)
        acc    = (preds == y_te).mean()
        macro_f1 = f1_score(y_te, preds, average='macro')
        print(f"\n  [{tag}] Acc={acc:.4f}, Macro F1={macro_f1:.4f}")
        print(classification_report(y_te, preds,
                                    target_names=['REAL', 'OldFake', 'NewFake'], digits=4))
        fname_tag = tag.split()[0].lower().replace('-', '')
        plot_confusion(y_te, preds, f'{tag} Confusion Matrix',
                       f'a5_{fname_tag}_confusion.png')
        results[tag] = {'accuracy': acc, 'macro_f1': macro_f1}

    # ── XGBoost (A-5 8d) ──────────────────────────────────
    X_tr = feat_a5[idx_tr]; X_te = feat_a5[idx_te]
    y_tr = y[idx_tr];        y_te = y[idx_te]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=SEED, n_jobs=-1
    )
    xgb_model.fit(X_tr_s, y_tr, eval_set=[(X_te_s, y_te)], verbose=False)
    preds_xgb = xgb_model.predict(X_te_s)
    acc_xgb    = (preds_xgb == y_te).mean()
    f1_xgb     = f1_score(y_te, preds_xgb, average='macro')
    print(f"\n  [A-5 XGBoost] Acc={acc_xgb:.4f}, Macro F1={f1_xgb:.4f}")
    print(classification_report(y_te, preds_xgb,
                                target_names=['REAL', 'OldFake', 'NewFake'], digits=4))
    plot_confusion(y_te, preds_xgb, 'A-5 XGBoost Confusion Matrix', 'a5_xgb_confusion.png')
    results['A-5 (8d, XGB)'] = {'accuracy': acc_xgb, 'macro_f1': f1_xgb}

    # ── 성능 비교 그래프 ───────────────────────────────────
    plot_performance_comparison(results, 'a5_comparison.png')

    print("\n✅ A-5 완료!\n")


if __name__ == "__main__":
    run()
