"""
A-2: Multi-band Energy Feature + SVM / XGBoost 3진 분류
- 피처: 4개 주파수 밴드 평균 에너지 벡터
- 분류: SVM (RBF), XGBoost
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import xgboost as xgb

from feature_extractor import extract_and_save

# ─── 밴드 경계 설정 ────────────────────────────────────────
BANDS = [(0, 50), (50, 150), (150, 250), (250, None)]
BAND_NAMES = ['Low (0-50)', 'Mid (50-150)', 'Mid-High (150-250)', 'High (250+)']
SEED  = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'freq_3class_outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_multiband_features(profiles):
    """[N, L] → [N, 4] 멀티밴드 에너지 벡터."""
    feats = []
    L = profiles.shape[1]
    for lo, hi in BANDS:
        hi_ = hi if hi is not None else L
        feats.append(profiles[:, lo:hi_].mean(axis=1, keepdims=True))
    return np.concatenate(feats, axis=1)   # [N, 4]


def plot_confusion(y_true, y_pred, title, fname):
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=['REAL', 'OldFake', 'NewFake'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 Confusion Matrix 저장: {path}")


def plot_pca_scatter(X_feats, y, title, fname):
    """PCA 2D 산점도."""
    pca   = PCA(n_components=2, random_state=SEED)
    X_2d  = pca.fit_transform(X_feats)
    names  = {0: 'REAL', 1: 'Old Fake', 2: 'New Fake'}
    colors = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}
    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl in [0, 1, 2]:
        mask = (y == lbl)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=names[lbl], color=colors[lbl], alpha=0.4, s=15)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 PCA 산점도 저장: {path}")


def plot_band_importance(importances, fname):
    """XGBoost 피처 중요도 막대그래프."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#90CAF9', '#64B5F6', '#42A5F5', '#1565C0']
    bars   = ax.bar(BAND_NAMES, importances, color=colors, edgecolor='black')
    for bar, val in zip(bars, importances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Feature Importance', fontsize=12)
    ax.set_title('A-2: XGBoost Feature Importance (Band별)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(importances) * 1.2)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 Feature Importance 저장: {path}")


def run():
    print("=" * 60)
    print("  A-2: Multi-band Energy + SVM / XGBoost")
    print("=" * 60)

    # ── 데이터 로드 ─────────────────────────────────────────
    data = extract_and_save()
    X_raw = data['profiles_orig']  # [N, L]
    y     = data['labels']

    # ── 피처 추출 ─────────────────────────────────────────
    X = extract_multiband_features(X_raw)  # [N, 4]
    print(f"\n  피처 shape: {X.shape}  (N={X.shape[0]}, 4-band)")

    # ── Train / Test 분할 ──────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # ── 스케일링 ──────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ══════════════════════════════════════════════════════
    # [1] SVM (RBF)
    # ══════════════════════════════════════════════════════
    print("\n  [SVM] 학습 중...")
    svm_params = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 0.01]}
    svm_cv = GridSearchCV(SVC(kernel='rbf', random_state=SEED),
                          svm_params, cv=5, scoring='f1_macro', n_jobs=-1, verbose=0)
    svm_cv.fit(X_tr_s, y_tr)
    best_svm = svm_cv.best_estimator_
    print(f"  SVM 최적 파라미터: {svm_cv.best_params_}")
    preds_svm = best_svm.predict(X_te_s)
    print("\n  [SVM Test 결과]")
    print(classification_report(y_te, preds_svm,
                                target_names=['REAL', 'OldFake', 'NewFake'], digits=4))
    plot_confusion(y_te, preds_svm, 'A-2 SVM Confusion Matrix', 'a2_svm_confusion.png')

    # ══════════════════════════════════════════════════════
    # [2] XGBoost
    # ══════════════════════════════════════════════════════
    print("\n  [XGBoost] 학습 중...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=SEED, n_jobs=-1
    )
    xgb_model.fit(X_tr_s, y_tr,
                  eval_set=[(X_te_s, y_te)], verbose=False)
    preds_xgb = xgb_model.predict(X_te_s)
    print("\n  [XGBoost Test 결과]")
    print(classification_report(y_te, preds_xgb,
                                target_names=['REAL', 'OldFake', 'NewFake'], digits=4))
    plot_confusion(y_te, preds_xgb, 'A-2 XGBoost Confusion Matrix', 'a2_xgb_confusion.png')
    plot_band_importance(xgb_model.feature_importances_, 'a2_feature_importance.png')

    # ── PCA 시각화 ────────────────────────────────────────
    plot_pca_scatter(X_tr_s, y_tr, 'A-2: PCA 2D (4-band features, Train)', 'a2_pca_scatter.png')

    print("\n✅ A-2 완료!\n")


if __name__ == "__main__":
    run()
