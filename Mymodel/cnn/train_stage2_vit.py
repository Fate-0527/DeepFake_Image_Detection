import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. 하이퍼파라미터 및 경로 설정
# ==========================================
DATASET_DIR = "/data1/DeepFake/dataset_cnn"
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.05
PATIENCE = 15          # Early Stopping patience
LABEL_SMOOTHING = 0.05 # Label Smoothing factor
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL_PATH = "v3_best_new_hunter_vit.pth"

NEW_FAKE_FOLDERS = ['flux_dev', 'nano_banana', 'std_3.5_large_turbo']
OLD_FAKE_FOLDERS = ['SD1_4', 'SD1_5', 'OpenJourney']

# ==========================================
# 2. 커스텀 Dataset 클래스 (ViT 규격에 맞게 Upsampling)
# ==========================================
class Stage2ViTDataset(Dataset):
    def __init__(self, filepaths, labels, is_train=False):
        self.filepaths = filepaths
        self.labels = labels
        self.is_train = is_train
        self.resize = transforms.Resize((224, 224), antialias=True)
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # 🌟 3채널 그대로 사용!
        tensor = torch.load(self.filepaths[idx])
        label = self.labels[idx]

        for c in range(3):
            mean = tensor[c].mean()
            std = tensor[c].std()
            tensor[c] = (tensor[c] - mean) / (std + 1e-8)
            
        tensor = self.resize(tensor) # [3, 224, 224]
        
        if self.is_train:
            tensor = self.augment(tensor)
            # Label Smoothing은 학습 시에만 적용
            smoothed_label = label * (1 - LABEL_SMOOTHING) + (1 - label) * LABEL_SMOOTHING
            return tensor, torch.tensor([smoothed_label], dtype=torch.float32)
        
        return tensor, torch.tensor([label], dtype=torch.float32)

# ==========================================
# 3. 모델 정의 (Vision Transformer)
# ==========================================
def get_model():
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 1)
    )
    return model.to(DEVICE)

# ==========================================
# 4. 메인 학습 파이프라인 (2단계 1:1 층화 추출)
# ==========================================
def train_and_validate():
    all_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.pt"), recursive=True)
    real_files = []
    new_fake_files = []
    
    for fpath in all_files:
        if any(old_fake in fpath for old_fake in OLD_FAKE_FOLDERS): continue
        if any(new_fake in fpath for new_fake in NEW_FAKE_FOLDERS):
            new_fake_files.append(fpath)
        elif "real" in fpath.lower():
            real_files.append(fpath)
            
    # 1:1 균형 맞추기
    target_count = len(new_fake_files)
    random.seed(42)
    balanced_real_files = random.sample(real_files, target_count)
    
    random.shuffle(balanced_real_files)
    random.shuffle(new_fake_files)
    
    # 8:2 층화 분할
    train_size = int(0.8 * target_count)
    
    train_files = balanced_real_files[:train_size] + new_fake_files[:train_size]
    train_labels = [0.0] * train_size + [1.0] * train_size
    
    val_files = balanced_real_files[train_size:] + new_fake_files[train_size:]
    val_labels = [0.0] * (target_count - train_size) + [1.0] * (target_count - train_size)

    print(f"\n📁 [Stage 2 - ViT] 완벽한 Stratified Split 분할 완료")
    print(f"   - Train Set: 총 {len(train_files)}장 (REAL {train_size} vs FAKE {train_size})")
    print(f"   - Val Set: 총 {len(val_files)}장 (REAL {target_count - train_size} vs FAKE {target_count - train_size})")

    train_dataset = Stage2ViTDataset(train_files, train_labels, is_train=True)
    val_dataset = Stage2ViTDataset(val_files, val_labels, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = get_model()
    # 1:1 비율이므로 별도 가중치 불필요
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print("\n🚀 [Step 2 - ViT] New Fake Hunter 학습 시작")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_dataset)
        scheduler.step()
        
        model.eval()
        val_loss, corrects = 0.0, 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) >= 0.5
                corrects += (preds == labels).sum().item()
                
        avg_val_loss = val_loss / len(val_dataset)
        val_acc = corrects / len(val_dataset)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"📈 Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # Best model 저장 (Val Acc 기준)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"  🌟 Best Model 저장됨! (Val Acc: {best_val_acc*100:.2f}%, Val Loss: {avg_val_loss:.4f})")
        
        # Early Stopping (Val Loss 기준 — 과적합 방지)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early Stopping! (Epoch {epoch+1}) — Val Loss가 {PATIENCE} epoch 동안 개선되지 않음")
                break

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Stage 2 (ViT) Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='purple')
    plt.title('Stage 2 (ViT) Accuracy')
    plt.legend()
    plt.savefig('stage2_vit_training_history_overfitting_solve.png', dpi=150)
    print(f"\n✅ 2단계 ViT 완료! 최고 정확도: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    train_and_validate()