import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import csv
import glob
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import torch.nn.functional as F

# WandB 끄기
os.environ["WANDB_MODE"] = "disabled"

# ---------------------------------------------------------
# 사용자 모듈 임포트
# ---------------------------------------------------------
from config import Config
from dataset import MasterReplicaDataset
from utils import set_seed, AverageMeter
from fire_model_binary import FIRE_model

# ---------------------------------------------------------
# [Helper Class] 경로(Path)를 반환해주는 데이터셋 래퍼
# ---------------------------------------------------------
class PathWrapperDataset(Dataset):
    def __init__(self, original_dataset, paths):
        self.dataset = original_dataset
        self.paths = paths
        assert len(self.dataset) == len(self.paths), "데이터셋과 경로 리스트의 길이가 다릅니다!"

    def __getitem__(self, index):
        img, label = self.dataset[index]
        path = self.paths[index]
        return img, label, path

    def __len__(self):
        return len(self.dataset)

# ---------------------------------------------------------
# [Helper Function] 경로 수집 함수 (Config 튜플 처리)
# ---------------------------------------------------------
def collect_paths_from_tuples(dir_config):
    all_paths = []
    for dir_path, slice_range in dir_config.items():
        files = glob.glob(os.path.join(dir_path, "*.*"))
        files.sort()
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        
        if not files:
            print(f"⚠️ Warning: No images found in {dir_path}")
            continue

        start, end = slice_range
        selected_files = files[start:end]
        all_paths.extend(selected_files)
    return all_paths

# ---------------------------------------------------------
# Loss Function Definitions
# ---------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        else: return focal_loss.sum()

class FIRELoss(nn.Module):
    def __init__(self, lambda0=0.2, lambda1=0.2, lambda2=0.6):
        super(FIRELoss, self).__init__()
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2 
        self.classifier_loss = FocalLoss(gamma=2.0) 
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_logits, labels, x_mid_latent, delta_x, m_mid, m_mid_c, i_mask, r_i_mask):
        loss_cls = self.classifier_loss(pred_logits, labels.float())
        loss_mid_rec = self.mse_loss(x_mid_latent, delta_x)
        loss_mask = self.mse_loss(m_mid, i_mask) + self.mse_loss(m_mid_c, r_i_mask) + \
                    torch.mean((1 - m_mid - m_mid_c) ** 2)
        
        total_loss = (self.lambda0 * loss_mid_rec) + \
                     (self.lambda1 * loss_mask) + \
                     (self.lambda2 * loss_cls)
        return total_loss, {"ce": loss_cls.item(), "mid_rec": loss_mid_rec.item(), "mask": loss_mask.item()}

# ---------------------------------------------------------
# [Analysis 1] 콘솔 출력용: 확장자별 성능 요약
# ---------------------------------------------------------
def evaluate_by_extension(all_labels, all_preds, all_paths):
    print("\n" + "="*50)
    print("📊 확장자별 상세 성능 분석 (Console Report)")
    print("="*50)
    
    groups = {'PNG': [], 'JPG': []}
    
    for i, path in enumerate(all_paths):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.png':
            groups['PNG'].append(i)
        elif ext in ['.jpg', '.jpeg']:
            groups['JPG'].append(i)
            
    for ext_name, indices in groups.items():
        if len(indices) == 0: continue
            
        sub_labels = np.array(all_labels)[indices]
        sub_preds = np.array(all_preds)[indices]
        
        acc = accuracy_score(sub_labels, sub_preds)
        prec = precision_score(sub_labels, sub_preds, zero_division=0)
        rec = recall_score(sub_labels, sub_preds, zero_division=0)
        f1 = f1_score(sub_labels, sub_preds, zero_division=0)
        
        print(f"🔹 [{ext_name}] (Count: {len(indices)})")
        print(f"   - Accuracy : {acc:.2%}")
        print(f"   - Precision: {prec:.4f}")
        print(f"   - Recall   : {rec:.4f}")
        print(f"   - F1-Score : {f1:.4f}")
        print("-" * 30)

# ---------------------------------------------------------
# [Analysis 2] 파일 저장용: 상세 분석 (Top-K일 때만 실행)
# ---------------------------------------------------------
def evaluate_detailed_metrics(targets, preds, paths, epoch, base_save_dir):
    detail_folder_name = getattr(Config, 'DETAIL_FILE', 'analysis_results')
    analysis_dir = os.path.join(base_save_dir, detail_folder_name)
    os.makedirs(analysis_dir, exist_ok=True)

    data = []
    for t, p, path in zip(targets, preds, paths):
        ext = os.path.splitext(path)[1].lower()
        ext_group = 'PNG' if ext == '.png' else ('JPG' if ext in ['.jpg', '.jpeg'] else 'Other')
        gen_name = 'Real' if t == 0 else os.path.basename(os.path.dirname(path))
        data.append({'Target': t, 'Pred': p, 'Extension': ext_group, 'Generator': gen_name})
    
    df = pd.DataFrame(data)

    # 1. Confusion Matrix
    for ext in ['PNG', 'JPG']:
        subset = df[df['Extension'] == ext]
        if len(subset) == 0: continue
        
        cm = confusion_matrix(subset['Target'], subset['Pred'], labels=[0, 1])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix ({ext}) - Epoch {epoch}')
        plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f"cm_{ext}_ep{epoch}.png"))
        plt.close()

    # 2. Generator별 모든 지표 (Acc, Prec, Rec, F1) 저장
    generators = sorted([g for g in df['Generator'].unique() if g != 'Real'])
    gen_metrics = []
    real_subset = df[df['Generator'] == 'Real']
    
    for gen in generators:
        gen_subset = df[df['Generator'] == gen]
        combined = pd.concat([real_subset, gen_subset])
        
        acc = accuracy_score(combined['Target'], combined['Pred'])
        prec = precision_score(combined['Target'], combined['Pred'], zero_division=0)
        rec = recall_score(combined['Target'], combined['Pred'], zero_division=0)
        f1 = f1_score(combined['Target'], combined['Pred'], zero_division=0)
        
        gen_metrics.append({'Generator': gen, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
        
    metrics_df = pd.DataFrame(gen_metrics)
    if not metrics_df.empty:
        # CSV 저장
        metrics_df.to_csv(os.path.join(analysis_dir, f"generator_metrics_ep{epoch}.csv"), index=False)
        
        # 그래프 저장 (F1 기준)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=metrics_df, x='Generator', y='F1-Score', palette='viridis')
        plt.title(f'F1 Score by Generator - Epoch {epoch}')
        plt.ylim(0, 1.0); plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f"generator_f1_ep{epoch}.png"))
        plt.close()

# ---------------------------------------------------------
# [Visualization] 학습 추이 그래프 (4분할: Loss, Acc, Prec/Rec, F1)
# ---------------------------------------------------------
def plot_metrics(history, save_dir):
    epochs = range(1, len(history['acc']) + 1)
    
    plt.figure(figsize=(12, 10)) # 2x2 사이즈

    # 1. Loss Trend
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['loss'], 'r-o', label='Val Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.grid(True, linestyle='--', alpha=0.7); plt.legend()

    # 2. Accuracy Trend
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['acc'], 'b-o', label='Val Acc')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.grid(True, linestyle='--', alpha=0.7); plt.legend()

    # 3. Precision & Recall Trend
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['precision'], 'g-o', label='Precision')
    plt.plot(epochs, history['recall'], 'c-o', label='Recall')
    plt.title('Precision & Recall')
    plt.xlabel('Epochs'); plt.ylabel('Score'); plt.grid(True, linestyle='--', alpha=0.7); plt.legend()

    # 4. F1-Score Trend
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['f1'], 'm-o', label='F1-Score')
    plt.title('F1-Score')
    plt.xlabel('Epochs'); plt.ylabel('Score'); plt.grid(True, linestyle='--', alpha=0.7); plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{Config.PNG_FILE}")
    plt.close()

# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------
def train_fire():
    set_seed(Config.RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f">>> Device: {device}")
    
    print(">>> 데이터 경로 수집 중...")
    train_real = collect_paths_from_tuples(Config.REAL_DIRS)
    train_fake = collect_paths_from_tuples(Config.FAKE_DIRS)
    test_real = collect_paths_from_tuples(Config.Vaild_REAL_DIRS)
    test_fake = collect_paths_from_tuples(Config.Vaild_FAKE_DIRS)
    
    print(f"📊 Dataset Summary:")
    print(f"   - Train Real: {len(train_real)} | Fake: {len(train_fake)}")
    print(f"   - Valid Real: {len(test_real)}  | Fake: {len(test_fake)}")
    
    train_ds = MasterReplicaDataset(train_real, train_fake, img_size=Config.IMG_SIZE, 
                                  use_master_replica=Config.USE_MASTER_REPLICA, 
                                  compression_prob=Config.COMPRESSION_PROB)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)

    test_ds_raw = MasterReplicaDataset(test_real, test_fake, img_size=Config.IMG_SIZE, mode='test')
    test_paths_ordered = test_real + test_fake 
    test_ds = PathWrapperDataset(test_ds_raw, test_paths_ordered)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    model = FIRE_model(radiuslow=Config.R_MIN, radiushigh=Config.R_MAX, device=device).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = FIRELoss()
    scaler = torch.cuda.amp.GradScaler()

    # Top-K 관리용 리스트: (val_loss, epoch, path) 튜플 저장
    top_k_checkpoints = [] 
    k_num = 3 
    history = {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []}
    
    log_file_path = f"{Config.SAVE_DIR}/training_log.csv"
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Epoch_Time(m)', 'Avg_Batch_Time(s)'])

    print(f">>> 학습 시작 (총 {Config.NUM_EPOCHS} Epochs)")
    
    for epoch in range(Config.NUM_EPOCHS):
        # === 학습 (Train) ===
        epoch_start_time = time.time()
        model.train()
        train_losses = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            labels_one_hot = F.one_hot(labels, num_classes=2).float()
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred_logits, x_mid_latent, delta_x, m_mid, m_mid_c = model(imgs)
                i_mask = model.fft_filter_module.i_mask
                r_i_mask = model.fft_filter_module.r_i_mask
                loss, _ = criterion(pred_logits, labels_one_hot, x_mid_latent, delta_x, m_mid, m_mid_c, i_mask, r_i_mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.update(loss.item(), imgs.size(0))
            pbar.set_postfix(loss=f"{train_losses.avg:.4f}")
        
        train_end = time.time()
        train_duration = train_end - epoch_start_time
        avg_batch_time = train_duration / len(train_loader)
        
        # === 검증 (Validation) ===
        model.eval()
        val_losses = AverageMeter()
        
        all_val_preds = []
        all_val_labels = []
        all_val_paths = []
        
        with torch.no_grad():
            for imgs, labels, paths in tqdm(test_loader, desc=f"Epoch {epoch+1} [Valid]", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                labels_one_hot = F.one_hot(labels, num_classes=2).float()
                
                with torch.cuda.amp.autocast():
                    pred_logits, x_mid_latent, delta_x, m_mid, m_mid_c = model(imgs)
                    i_mask = model.fft_filter_module.i_mask
                    r_i_mask = model.fft_filter_module.r_i_mask
                    loss, _ = criterion(pred_logits, labels_one_hot, x_mid_latent, delta_x, m_mid, m_mid_c, i_mask, r_i_mask)
                    val_losses.update(loss.item(), imgs.size(0))
                
                preds_cls = torch.argmax(pred_logits, dim=1).cpu().numpy().flatten()
                all_val_preds.extend(preds_cls)
                all_val_labels.extend(labels.cpu().numpy().flatten())
                all_val_paths.extend(paths)

        # 1. 전체 지표 계산
        avg_val_loss = val_losses.avg
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_pre = precision_score(all_val_labels, all_val_preds, zero_division=0)
        val_rec = recall_score(all_val_labels, all_val_preds, zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)
        
        epoch_min = train_duration / 60.0
        print(f"\n[Epoch {epoch+1}] Train Loss: {train_losses.avg:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2%}")
        
        # 콘솔에 상세 성능 출력
        evaluate_by_extension(all_val_labels, all_val_preds, all_val_paths)

        # -----------------------------------------------------------
        # ★ [핵심] Top-K 판단 및 파일 관리 (F1 기준: 높을수록 Good)
        # -----------------------------------------------------------
        is_top_k = False
        if len(top_k_checkpoints) < k_num:
            is_top_k = True
        elif val_f1 > top_k_checkpoints[-1][0]: # 현재 F1이 Worst(3등)보다 크면
            is_top_k = True

        if is_top_k:
            print(f"🌟 Top-{k_num} Epoch 달성! (F1: {val_f1:.4f}) 상세 분석 파일 생성 중...")
            
            # (1) 상세 분석 파일 생성
            evaluate_detailed_metrics(all_val_labels, all_val_preds, all_val_paths, epoch+1, Config.SAVE_DIR)
            
            # (2) Checkpoint 저장
            current_filename = f"{Config.FILE_NAME}_ep{epoch+1}_f1{val_f1:.4f}.pth"
            current_ckpt_path = os.path.join(Config.SAVE_DIR, current_filename)
            torch.save(model.state_dict(), current_ckpt_path)
            
            # (3) 리스트 업데이트 (F1 내림차순 정렬: [0]이 best, [-1]이 worst)
            top_k_checkpoints.append((val_f1, epoch+1, current_ckpt_path))
            top_k_checkpoints.sort(key=lambda x: x[0], reverse=True) 
            
            # (4) 밀려난(Worst) 에포크 파일 삭제
            if len(top_k_checkpoints) > k_num:
                worst = top_k_checkpoints.pop() # (f1, epoch, path)
                worst_epoch = worst[1]
                
                # 모델 삭제
                if os.path.exists(worst[2]):
                    try: os.remove(worst[2]); print(f">>> 🗑️ 모델 삭제: Epoch {worst_epoch}")
                    except: pass
                
                # 분석 파일 삭제 (*_ep{worst_epoch}.*)
                detail_folder = getattr(Config, 'DETAIL_FILE', 'analysis_results')
                analysis_dir = os.path.join(Config.SAVE_DIR, detail_folder)
                if os.path.exists(analysis_dir):
                    files_to_delete = glob.glob(os.path.join(analysis_dir, f"*_ep{worst_epoch}.*"))
                    for f in files_to_delete:
                        try: os.remove(f)
                        except: pass
                    if files_to_delete: print(f">>> 🗑️ 분석 파일 {len(files_to_delete)}개 삭제 (Epoch {worst_epoch})")

        # [Latest] 저장
        latest_path = os.path.join(Config.SAVE_DIR, f"{Config.FILE_NAME}_latest.pth")
        torch.save(model.state_dict(), latest_path)

        # History & Plot (매번 그림)
        history['loss'].append(avg_val_loss)
        history['acc'].append(val_acc)
        history['precision'].append(val_pre)
        history['recall'].append(val_rec)
        history['f1'].append(val_f1)
        plot_metrics(history, Config.SAVE_DIR)
        
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_losses.avg, avg_val_loss, val_acc, val_pre, val_rec, val_f1, f"{epoch_min:.2f}", f"{avg_batch_time:.4f}"])
        
        scheduler.step(avg_val_loss)
        plt.close('all')

if __name__ == "__main__":
    train_fire()