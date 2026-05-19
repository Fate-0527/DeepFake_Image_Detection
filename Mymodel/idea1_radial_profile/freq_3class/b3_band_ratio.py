"""
B-3: 주파수 대역별 비율(Band Ratio) 분석
- 전체 주파수를 LF / MF / HF 세 대역으로 나눠 에너지 합산
- 비율(Ratio)을 Feature로 사용 → 절대 밝기 영향 제거
- Real vs Old Fake vs New Fake의 대역 비율 차이 시각화 + SVM 분류
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import f_oneway

from feature_extractor import extract_and_save, LABEL_REAL

SEED = 42

# 대역 경계 (반경 단위)
LF_END  =  80    # Low  Frequency:   0 ~ 80
MF_END  = 250    # Mid  Frequency:  80 ~ 250
# HF:  250 ~ end

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'freq_analysis_outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_band_ratios(profiles):
    """
    각 이미지의 방사형 프로파일을 3개 대역으로 나눠 에너지 비율 계산.
    반환: [N, 7]
      - lf_energy, mf_energy, hf_energy (log scale sum)
      - hf_lf_ratio, hf_mf_ratio, mf_lf_ratio
      - hf_total_ratio (hf / 전체 에너지)
    """
    lf = profiles[:, :LF_END]
    mf = profiles[:, LF_END:MF_END]
    hf = profiles[:, MF_END:]

    # log magnitude이므로 합산이 에너지 대표값
    lf_e = lf.sum(axis=1)
    mf_e = mf.sum(axis=1)
    hf_e = hf.sum(axis=1)
    total = lf_e + mf_e + hf_e + 1e-10

    return np.stack([
        lf_e,
        mf_e,
        hf_e,
        hf_e / (lf_e + 1e-10),      # HF / LF
        hf_e / (mf_e + 1e-10),      # HF / MF
        mf_e / (lf_e + 1e-10),      # MF / LF
        hf_e / total,               # HF 비율
    ], axis=1)


def run():
    print("=" * 60)
    print("  B-3: 주파수 대역별 비율(Band Ratio) 분석")
    print("=" * 60)

    data  = extract_and_save()
    X_raw = data['profiles_orig']
    y     = data['labels']
    L     = X_raw.shape[1]
    print(f"  Profile 길이: {L}  |  HF 시작: r={MF_END}")

    # ── 대역별 에너지 곡선 시각화 ─────────────────────────────
    names  = {0: 'REAL', 1: 'Old Fake', 2: 'New Fake'}
    colors = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(14, 6))
    for lbl in [0, 1, 2]:
        r  = X_raw[y == lbl]
        mu  = r.mean(axis=0)
        std = r.std(axis=0)
        ax.plot(x, mu, label=names[lbl], color=colors[lbl], lw=2)
        ax.fill_between(x, mu - std, mu + std, color=colors[lbl], alpha=0.12)

    # 대역 경계 표시
    ax.axvspan(0,     LF_END, alpha=0.06, color='blue',   label=f'LF (0~{LF_END})')
    ax.axvspan(LF_END, MF_END, alpha=0.06, color='orange', label=f'MF ({LF_END}~{MF_END})')
    ax.axvspan(MF_END, L,     alpha=0.06, color='red',    label=f'HF ({MF_END}~)')
    ax.axvline(LF_END, color='blue',   ls='--', lw=1.2)
    ax.axvline(MF_END, color='red',    ls='--', lw=1.2)

    ax.set_xlabel('Frequency Radius (r)', fontsize=12)
    ax.set_ylabel('Log Magnitude', fontsize=12)
    ax.set_title('B-3: 주파수 대역별 에너지 분포 (LF / MF / HF)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, 'b3_band_energy_curves.png')
    plt.savefig(path1, dpi=150); plt.close()
    print(f"  📊 대역 에너지 곡선 저장: {path1}")

    # ── Band Ratio Feature 계산 ───────────────────────────
    feats = compute_band_ratios(X_raw)
    feat_labels = ['LF_e', 'MF_e', 'HF_e', 'HF/LF', 'HF/MF', 'MF/LF', 'HF_ratio']

    # ── 시각화 2: 클래스별 Ratio 박스플롯 ──────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, (fname_f, ax) in enumerate(zip(feat_labels, axes)):
        data_by_class = [feats[y == lbl, i] for lbl in [0, 1, 2]]
        bp = ax.boxplot(data_by_class, labels=['REAL', 'OldFake', 'NewFake'],
                        patch_artist=True,
                        boxprops=dict(facecolor='lightgray'),
                        medianprops=dict(color='red', lw=2))
        for patch, color in zip(bp['boxes'], ['#2196F3', '#F44336', '#4CAF50']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # ANOVA p-value
        fval, pval = f_oneway(*data_by_class)
        ax.set_title(f'{fname_f}\n(p={pval:.2e})', fontsize=10, fontweight='bold',
                     color='darkred' if pval < 0.001 else 'black')
        ax.grid(True, alpha=0.3)
    axes[-1].axis('off')
    plt.suptitle('B-3: 주파수 대역 비율별 클래스 분포 (ANOVA p-value 표시)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, 'b3_band_ratio_boxplot.png')
    plt.savefig(path2, dpi=150); plt.close()
    print(f"  📊 Band Ratio 박스플롯 저장: {path2}")

    # ── 시각화 3: HF/LF vs HF/MF 산점도 (클래스 분포) ────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl in [0, 1, 2]:
        mask = y == lbl
        ax.scatter(feats[mask, 3], feats[mask, 4],
                   label=names[lbl], color=colors[lbl], alpha=0.3, s=12)
    ax.set_xlabel('HF / LF Ratio', fontsize=11)
    ax.set_ylabel('HF / MF Ratio', fontsize=11)
    ax.set_title('B-3: HF/LF vs HF/MF 산점도', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, 'b3_scatter_hf_ratio.png')
    plt.savefig(path3, dpi=150); plt.close()
    print(f"  📊 산점도 저장: {path3}")

    # ── 분류: SVM ─────────────────────────────────────────
    print("\n  [SVM 분류]")
    X_tr, X_te, y_tr, y_te = train_test_split(feats, y, test_size=0.2, random_state=SEED, stratify=y)
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
    ax.set_title('B-3 Band Ratio SVM Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path4 = os.path.join(OUTPUT_DIR, 'b3_svm_confusion.png')
    plt.savefig(path4, dpi=150); plt.close()
    print(f"  📊 Confusion Matrix 저장: {path4}")

    print("\n✅ B-3 완료!\n")


if __name__ == "__main__":
    run()
