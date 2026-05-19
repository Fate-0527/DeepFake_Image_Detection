import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import random
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 기존 유틸 및 설정 임포트
from utils.network_utils import FIRE_model 
from fire_dataset_adapter import CustomFIREDataset 
from config import Config 

# --------------------------------------------------------------------------------
# [Helper Class] 데이터셋에서 경로(Path)를 함께 반환하도록 감싸는 래퍼 클래스
# --------------------------------------------------------------------------------
class PathWrapperDataset(Dataset):
    def __init__(self, original_dataset, paths):
        self.dataset = original_dataset
        self.paths = paths
        assert len(self.dataset) == len(self.paths), f"Dataset len ({len(self.dataset)}) != Paths len ({len(self.paths)})"

    def __getitem__(self, index):
        img, label = self.dataset[index]
        path = self.paths[index]
        return img, label, path

    def __len__(self):
        return len(self.dataset)

# --------------------------------------------------------------------------------
# [Helper Function] 경로 수집 함수
# --------------------------------------------------------------------------------
def collect_paths_with_slicing(dir_config):
    all_paths = []
    exts = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    
    for dir_path, (start_idx, end_idx) in dir_config.items():
        if not os.path.exists(dir_path):
            print(f"⚠️ Warning: Directory not found -> {dir_path}")
            continue
            
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(dir_path, ext)))
            files.extend(glob.glob(os.path.join(dir_path, ext.upper())))
            
        files.sort()
        selected_files = files[start_idx:end_idx]
        all_paths.extend(selected_files)
        
    return all_paths

# --------------------------------------------------------------------------------
# [Analysis Function] 상세 분석 및 시각화 (Config.SAVE_DIR 사용)
# --------------------------------------------------------------------------------
def evaluate_detailed_metrics(targets, preds, paths, epoch, base_save_dir):
    """
    base_save_dir (Config.SAVE_DIR) 아래에 analysis_results 폴더를 만들고 저장
    """
    # 1. 데이터 프레임 생성
    data = []
    for t, p, path in zip(targets, preds, paths):
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            ext_group = 'JPG'
        elif ext in ['.png']:
            ext_group = 'PNG'
        else:
            ext_group = 'Others'
            
        if t == 0: # Real Label
            gen_name = 'Real'
        else:
            # 폴더 구조 가정: .../FAKE/모델명/이미지.png
            gen_name = os.path.basename(os.path.dirname(path))
            
        data.append({
            'Target': t,
            'Pred': p,
            'Extension': ext_group,
            'Generator': gen_name
        })
    
    df = pd.DataFrame(data)
    
    # [수정] 저장 경로: Config.RESULT_DIR/analysis_results
    analysis_dir = os.path.join(base_save_dir, f"{Config.DETAIL_FILE}")
    os.makedirs(analysis_dir, exist_ok=True)

    # ==========================================
    # Task 1: 확장자별 Confusion Matrix (PNG vs JPG)
    # ==========================================
    for ext in ['PNG', 'JPG']:
        subset = df[df['Extension'] == ext]
        if len(subset) == 0:
            continue
            
        cm = confusion_matrix(subset['Target'], subset['Pred'], labels=[0, 1])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix ({ext}) - Epoch {epoch}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f"cm_{ext}_ep{epoch}.png"))
        plt.close()

    # ==========================================
    # Task 2: 생성 모델별 지표 (Acc, Prec, Recall, F1)
    # ==========================================
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
        
        gen_metrics.append({
            'Generator': gen,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Count': len(gen_subset)
        })
        
    metrics_df = pd.DataFrame(gen_metrics)
    
    # CSV 및 그래프 저장 (Config.SAVE_DIR/analysis_results/...)
    if not metrics_df.empty:
        csv_path = os.path.join(analysis_dir, f"generator_metrics_ep{epoch}.csv")
        metrics_df.to_csv(csv_path, index=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=metrics_df, x='Generator', y='F1-Score', palette='viridis')
        plt.title(f'F1 Score by Generator - Epoch {epoch}')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f"generator_f1_ep{epoch}.png"))
        plt.close()

# [함수] 전체 학습 추이 그래프 (기존 유지 - Config.PNG_DIR 사용)
def plot_training_results(history):
    save_dir = Config.RESULT_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, Config.PNG_FILE)

    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss'); plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Accuracy'); plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['val_precision'], 'g-', label='Val Prec')
    plt.plot(epochs, history['val_recall'], 'm-', label='Val Recall')
    plt.title('Precision & Recall'); plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['val_f1'], 'k-', label='Val F1')
    plt.title('F1 Score'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --------------------------------------------------------------------------------
# Main Function
# --------------------------------------------------------------------------------
def main():
    random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Config.SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    print(f">>> Device: {device}")
    print(f">>> Save Dir (Checkpoints & Analysis): {save_dir}")
    
    # 1. 데이터 경로 수집
    print(">>> Collecting File Paths...")
    train_real = collect_paths_with_slicing(Config.REAL_DIRS)
    train_fake = collect_paths_with_slicing(Config.FAKE_DIRS)
    val_real = collect_paths_with_slicing(Config.Vaild_REAL_DIRS)
    val_fake = collect_paths_with_slicing(Config.Vaild_FAKE_DIRS)
    
    random.shuffle(train_real)
    random.shuffle(train_fake)
    
    # Validation 경로 순서 정렬 (Real -> Fake 순)
    val_paths_ordered = val_real + val_fake 
    
    # 2. 데이터셋 생성
    train_ds_raw = CustomFIREDataset(train_real, train_fake, is_train=True)
    val_ds_raw = CustomFIREDataset(val_real, val_fake, is_train=False)
    
    # [Wrapper 적용] Path 추적용
    val_ds = PathWrapperDataset(val_ds_raw, val_paths_ordered)
    
    train_loader = DataLoader(train_ds_raw, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    print(f"📊 Dataset: Train({len(train_ds_raw)}) / Val({len(val_ds)})")

    # 3. 모델 설정
    model = FIRE_model(mode="frq").to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.002)

    top_k = 3
    saved_checkpoints = [] 
    best_val_loss = float('inf')

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []}
    
    print(">>> Start Training")
    
    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()
        
        # --- [Train Loop] ---
        model.train()
        train_loss_sum = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            out, middle_freq_image, raw_rec_delta, mask_mid, mask_filtered = model(imgs)
            
            loss_mse_rec = F.mse_loss(middle_freq_image, raw_rec_delta)
            
            i_mask = model.fft_filter_module.i_mask.unsqueeze(0).repeat(imgs.shape[0], 1, 1, 1).detach()
            r_i_mask = model.fft_filter_module.r_i_mask.unsqueeze(0).repeat(imgs.shape[0], 1, 1, 1).detach()
            all_mask = torch.ones_like(r_i_mask).detach()
            
            loss_mse_mask = F.mse_loss(mask_mid, i_mask) + F.mse_loss(mask_filtered, r_i_mask) + F.mse_loss(mask_mid + mask_filtered, all_mask)
            loss_b = F.binary_cross_entropy_with_logits(out[:, 0], labels.float())
            total_loss = 0.6 * loss_b + 0.2 * loss_mse_rec + 0.2 * loss_mse_mask
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss_sum += total_loss.item()
            probs = torch.sigmoid(out[:, 0])
            preds = (probs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}", 'acc': f"{train_correct/train_total:.2%}"})
            
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_acc = train_correct / train_total
        epoch_time = time.time() - epoch_start
        
        # --- [Validation Loop] ---
        model.eval()
        val_loss_sum = 0
        
        all_val_targets = []
        all_val_preds = []
        all_val_paths = []
        
        with torch.no_grad():
            for imgs, labels, paths in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                
                out, middle_freq_image, raw_rec_delta, mask_mid, mask_filtered = model(imgs)
                
                # Loss
                loss_mse_rec = F.mse_loss(middle_freq_image, raw_rec_delta)
                i_mask = model.fft_filter_module.i_mask.unsqueeze(0).repeat(imgs.shape[0], 1, 1, 1).detach()
                r_i_mask = model.fft_filter_module.r_i_mask.unsqueeze(0).repeat(imgs.shape[0], 1, 1, 1).detach()
                all_mask = torch.ones_like(r_i_mask).detach()
                loss_mse_mask = F.mse_loss(mask_mid, i_mask) + F.mse_loss(mask_filtered, r_i_mask) + F.mse_loss(mask_mid + mask_filtered, all_mask)
                loss_b = F.binary_cross_entropy_with_logits(out[:, 0], labels.float())
                val_loss = 0.6 * loss_b + 0.2 * loss_mse_rec + 0.2 * loss_mse_mask
                
                val_loss_sum += val_loss.item()
                probs = torch.sigmoid(out[:, 0])
                preds = (probs > 0.5).float()
                
                all_val_targets.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())
                all_val_paths.extend(paths)

        avg_val_loss = val_loss_sum / len(val_loader)
        
        # 전체 지표
        val_acc = accuracy_score(all_val_targets, all_val_preds)
        val_prec = precision_score(all_val_targets, all_val_preds, zero_division=0)
        val_rec = recall_score(all_val_targets, all_val_preds, zero_division=0)
        val_f1 = f1_score(all_val_targets, all_val_preds, zero_division=0)
        
        # ★ 상세 분석: Config.RESULT_DIR를 인자로 전달
        evaluate_detailed_metrics(all_val_targets, all_val_preds, all_val_paths, epoch+1, Config.RESULT_DIR)

        # History 기록 & 그래프 (여긴 Config.RESULT_DIR 사용)
        history['train_loss'].append(avg_train_loss); history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss); history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec); history['val_recall'].append(val_rec); history['val_f1'].append(val_f1)
        plot_training_results(history)
        
        print(f"⏱️  Time: {epoch_time:.1f}s")
        print(f"📊 Train Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.2%}")
        print(f"📊 Val   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.4f}")

        scheduler.step(avg_val_loss)

        # Top-k 저장 (Config.SAVE_DIR 사용)
        current_ckpt_name = f"{Config.FILE_NAME}_ep{epoch+1:03d}_val{avg_val_loss:.4f}.pth"
        current_ckpt_path = os.path.join(Config.SAVE_DIR, current_ckpt_name)
        torch.save(model.state_dict(), current_ckpt_path)
        saved_checkpoints.append((avg_val_loss, epoch+1, current_ckpt_path))
        saved_checkpoints.sort(key=lambda x: x[0])
        if len(saved_checkpoints) > top_k:
            worst = saved_checkpoints.pop()
            if os.path.exists(worst[2]): os.remove(worst[2])

        if avg_val_loss < best_val_loss:
            print(f"✅ Best Updated! ({best_val_loss:.4f} -> {avg_val_loss:.4f})")
            best_val_loss = avg_val_loss
        else:
            print(f"⚠️ No Improvement (Best: {best_val_loss:.4f})")

if __name__ == "__main__":
    main()