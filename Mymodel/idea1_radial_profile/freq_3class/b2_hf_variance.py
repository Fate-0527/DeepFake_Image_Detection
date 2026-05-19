"""
B-2: 고주파 대역 분산(Variance) 분석
- 고주파 반경(HF_LOW 이상)을 작은 Bin으로 나눠 각 Bin 내 Range/Variance 계산
- "고주파가 얼마나 울퉁불퉁한가" 를 기준으로 3진 분류
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from feature_extractor import extract_and_save, LABEL_REAL

SEED    = 42
HF_LOW  = 150          # 고주파 시작 반경
BIN_W   = 10           # 각 Bin 너비 (단위: 반경)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'freq_analysis_outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_bin_stats(profiles, hf_low=HF_LOW, bin_w=BIN_W):
    """
    고주파 대역(hf_low:)을 bin_w 단위로 나눠 각 Bin의
    mean, std, range(max-min), variance를 계산.
    반환: [N, n_bins * 4]
    """
    hf = profiles[:, hf_low:]
    n_bins = hf.shape[1] // bin_w
    hf = hf[:, :n_bins * bin_w].reshape(hf.shape[0], n_bins, bin_w)
    means = hf.mean(axis=2)         # [N, n_bins]
    stds  = hf.std(axis=2)
    ranges = hf.max(axis=2) - hf.min(axis=2)
    vars_  = hf.var(axis=2)
    return np.hstack([means, stds, ranges, vars_])   # [N, n_bins*4]


def run():
    print("=" * 60)
    print("  B-2: 고주파 대역 분산(Variance) 분석")
    print("=" * 60)

    data  = extract_and_save()
    X_raw = data['profiles_orig']   # [N, L]
    y     = data['labels']
    L     = X_raw.shape[1]
    print(f"  Profile 길이: {L}  |  샘플 수: {len(y)}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, random_state=SEED, stratify=y)

    # ── 시각화 1: 고주파 Bin Range 곡선 (클래스별 mean) ────
    hf      = X_raw[:, HF_LOW:]
    n_bins  = hf.shape[1] // BIN_W
    hf_cut  = hf[:, :n_bins * BIN_W].reshape(hf.shape[0], n_bins, BIN_W)
    bin_mid = HF_LOW + np.arange(n_bins) * BIN_W + BIN_W // 2

    names  = {0: 'REAL', 1: 'Old Fake', 2: 'New Fake'}
    colors = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 좌: Bin 내 Range (max-min)
    ax = axes[0]
    for lbl in [0, 1, 2]:
        r = hf_cut[y == lbl]
        bin_range = (r.max(axis=2) - r.min(axis=2)).mean(axis=0)
        ax.plot(bin_mid, bin_range, label=names[lbl], color=colors[lbl], lw=2)
    ax.set_xlabel('Frequency Radius (r)', fontsize=11)
    ax.set_ylabel('Mean Range within Bin', fontsize=11)
    ax.set_title('고주파 Bin Range (뾰족함 지표)', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 우: Bin 내 Variance
    ax = axes[1]
    for lbl in [0, 1, 2]:
        r = hf_cut[y == lbl]
        bin_var = r.var(axis=2).mean(axis=0)
        ax.plot(bin_mid, bin_var, label=names[lbl], color=colors[lbl], lw=2)
    ax.set_xlabel('Frequency Radius (r)', fontsize=11)
    ax.set_ylabel('Mean Variance within Bin', fontsize=11)
    ax.set_title('고주파 Bin Variance (울퉁불퉁 정도)', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('B-2: 고주파 대역 Bin별 분산 분석 (HF_LOW={})'.format(HF_LOW),
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, 'b2_hf_bin_variance.png')
    plt.savefig(path1, dpi=150); plt.close()
    print(f"  📊 Bin 분산 곡선 저장: {path1}")

    # ── 시각화 2: Real 대비 Residual (Bin Range 기준) ────────
    real_bin_range = (hf_cut[y == 0].max(axis=2) - hf_cut[y == 0].min(axis=2)).mean(axis=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    for lbl in [1, 2]:
        r = hf_cut[y == lbl]
        fake_range = (r.max(axis=2) - r.min(axis=2)).mean(axis=0)
        residual   = fake_range - real_bin_range
        ax.plot(bin_mid, residual, label=f'{names[lbl]} - Real', color=colors[lbl], lw=2)
        ax.fill_between(bin_mid, 0, residual, color=colors[lbl], alpha=0.15)
    ax.axhline(0, color='black', ls='-', lw=0.8)
    ax.set_xlabel('Frequency Radius (r)', fontsize=11)
    ax.set_ylabel('ΔRange (Fake - Real)', fontsize=11)
    ax.set_title('B-2: 고주파 Bin Range Residual (Real 대비 얼마나 더 뾰족한가)', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, 'b2_hf_residual.png')
    plt.savefig(path2, dpi=150); plt.close()
    print(f"  📊 Residual 곡선 저장: {path2}")

    # ── 분류 ─────────────────────────────────────────────
    feat_tr = compute_bin_stats(X_tr)
    feat_te = compute_bin_stats(X_te)
    sc = StandardScaler()
    feat_tr_s = sc.fit_transform(feat_tr)
    feat_te_s  = sc.transform(feat_te)

    # SVM
    print("\n  [SVM 분류]")
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
    svm.fit(feat_tr_s, y_tr)
    preds_svm = svm.predict(feat_te_s)
    print(classification_report(y_te, preds_svm, target_names=['REAL', 'OldFake', 'NewFake'], digits=4))

    # Random Forest (Feature Importance도 볼 수 있음)
    print("  [Random Forest 분류]")
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(feat_tr_s, y_tr)
    preds_rf = rf.predict(feat_te_s)
    print(classification_report(y_te, preds_rf, target_names=['REAL', 'OldFake', 'NewFake'], digits=4))

    # Feature Importance 시각화
    n_bins_feat = feat_tr.shape[1] // 4
    feat_labels = (
        [f'mean_bin{i}' for i in range(n_bins_feat)] +
        [f'std_bin{i}'  for i in range(n_bins_feat)] +
        [f'range_bin{i}' for i in range(n_bins_feat)] +
        [f'var_bin{i}'  for i in range(n_bins_feat)]
    )
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(20), importances[top_idx], color='steelblue', alpha=0.8)
    ax.set_xticks(range(20))
    ax.set_xticklabels([feat_labels[i] for i in top_idx], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Feature Importance', fontsize=11)
    ax.set_title('B-2: Random Forest Feature Importance (Top 20)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, 'b2_feature_importance.png')
    plt.savefig(path3, dpi=150); plt.close()
    print(f"  📊 Feature Importance 저장: {path3}")

    # Confusion Matrix (SVM)
    cm = confusion_matrix(y_te, preds_svm)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['REAL', 'OldFake', 'NewFake']).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('B-2 SVM Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path4 = os.path.join(OUTPUT_DIR, 'b2_svm_confusion.png')
    plt.savefig(path4, dpi=150); plt.close()
    print(f"  📊 Confusion Matrix 저장: {path4}")

    print("\n✅ B-2 완료!\n")


if __name__ == "__main__":
    run()
