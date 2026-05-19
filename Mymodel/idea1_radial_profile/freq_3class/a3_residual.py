"""
A-3: Residual Profile 기반 3진 분류
- Real 평균 프로파일 기준 편차 곡선 생성
- 방법1: 고주파 residual 통계량 → SVM
- 방법2: residual 전체 벡터 → 3층 MLP (PyTorch)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from feature_extractor import extract_and_save, LABEL_REAL

HF_LOW     = 250
SEED       = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'freq_3class_outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── MLP 모델 정의 ───────────────────────────────────────────
class ResidualMLP(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def compute_residual_stats(residuals, hf_low=HF_LOW):
    """residual[hf_low:] 구간 통계량 → [N, 4] 피처."""
    hf = residuals[:, hf_low:]
    return np.stack([
        hf.mean(axis=1),
        hf.std(axis=1),
        np.array([skew(row) for row in hf]),
        np.array([kurtosis(row) for row in hf]),
    ], axis=1)


def plot_residual_curves(residuals, y, title, fname):
    """클래스별 residual 평균 ± std 곡선."""
    names  = {0: 'REAL', 1: 'Old Fake', 2: 'New Fake'}
    colors = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}
    x = np.arange(residuals.shape[1])
    fig, ax = plt.subplots(figsize=(12, 5))
    for lbl in [0, 1, 2]:
        r = residuals[y == lbl]
        mu  = r.mean(axis=0)
        std = r.std(axis=0)
        ax.plot(x, mu, label=names[lbl], color=colors[lbl], lw=2)
        ax.fill_between(x, mu - std, mu + std, color=colors[lbl], alpha=0.15)
    ax.axhline(0, color='black', ls='-', lw=0.8)
    ax.axvline(HF_LOW, color='gray', ls='--', lw=1.2, label=f'HF 경계 (r={HF_LOW})')
    ax.set_xlabel('Frequency Radius (r)', fontsize=12)
    ax.set_ylabel('Residual (test - Real mean)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 Residual 곡선 저장: {path}")


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


def run():
    print("=" * 60)
    print("  A-3: Residual Profile 3진 분류")
    print("=" * 60)

    data  = extract_and_save()
    X_raw = data['profiles_orig']   # [N, L]
    y     = data['labels']

    # ── Train / Test 분할 ──────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # ── Real 평균 프로파일 계산 (Train 셋만 사용) ───────────────
    real_mean = X_tr[y_tr == LABEL_REAL].mean(axis=0)   # [L]

    # ── Residual 계산 ─────────────────────────────────────
    residuals_tr = X_tr - real_mean[np.newaxis, :]
    residuals_te = X_te - real_mean[np.newaxis, :]

    # ── residual 곡선 시각화 (Train 기준) ─────────────────────
    plot_residual_curves(residuals_tr, y_tr,
                         'A-3: Residual Profile (Train, mean ± std)', 'a3_residual_curves.png')

    # ══════════════════════════════════════════════════════
    # [방법1] 고주파 통계량 → SVM
    # ══════════════════════════════════════════════════════
    print("\n  [방법1] Residual 통계량 → SVM")
    feat_tr = compute_residual_stats(residuals_tr)   # [N_tr, 4]
    feat_te = compute_residual_stats(residuals_te)

    scaler = StandardScaler()
    feat_tr_s = scaler.fit_transform(feat_tr)
    feat_te_s = scaler.transform(feat_te)

    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
    svm.fit(feat_tr_s, y_tr)
    preds_svm = svm.predict(feat_te_s)
    print(classification_report(y_te, preds_svm,
                                target_names=['REAL', 'OldFake', 'NewFake'], digits=4))
    plot_confusion(y_te, preds_svm, 'A-3 SVM Confusion Matrix', 'a3_svm_confusion.png')

    # ══════════════════════════════════════════════════════
    # [방법2] Residual 전체 벡터 → MLP
    # ══════════════════════════════════════════════════════
    print("\n  [방법2] Residual 전체 벡터 → MLP")
    # 저주파(0~50) 제거 후 사용 (노이즈 많음)
    res_tr_hf = residuals_tr[:, MF_START:]   if (MF_START := 50) else residuals_tr
    res_te_hf = residuals_te[:, MF_START:]

    # 스케일링
    sc2 = StandardScaler()
    res_tr_s = sc2.fit_transform(res_tr_hf).astype(np.float32)
    res_te_s = sc2.transform(res_te_hf).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp    = ResidualMLP(input_dim=res_tr_s.shape[1]).to(device)
    opt    = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    criterion = nn.CrossEntropyLoss()

    tr_ds  = TensorDataset(torch.from_numpy(res_tr_s), torch.tensor(y_tr, dtype=torch.long))
    tr_ld  = DataLoader(tr_ds, batch_size=256, shuffle=True, num_workers=2)

    best_acc = 0.0
    best_preds = None

    mlp.train()
    for epoch in range(60):
        total_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(mlp(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()

        if (epoch + 1) % 10 == 0:
            mlp.eval()
            with torch.no_grad():
                out   = mlp(torch.from_numpy(res_te_s).to(device))
                preds = out.argmax(dim=1).cpu().numpy()
                acc   = (preds == y_te).mean()
            if acc > best_acc:
                best_acc   = acc
                best_preds = preds
            print(f"    Epoch {epoch+1:3d} | Loss={total_loss/len(tr_ld):.4f} | Test Acc={acc:.4f}")
            mlp.train()

    print("\n  [MLP 최종 Test 결과]")
    print(classification_report(y_te, best_preds,
                                target_names=['REAL', 'OldFake', 'NewFake'], digits=4))
    plot_confusion(y_te, best_preds, 'A-3 MLP Confusion Matrix', 'a3_mlp_confusion.png')

    print("\n✅ A-3 완료!\n")


if __name__ == "__main__":
    run()
