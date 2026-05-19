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
BATCH_SIZE = 64  # ViT 메모리 이슈 대비 축소
EPOCHS = 100
LEARNING_RATE = 3e-5 # ViT 맞춤형 저속 학습률
WEIGHT_DECAY = 0.05
PATIENCE = 15          # Early Stopping patience
LABEL_SMOOTHING = 0.05 # Label Smoothing factor
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL_PATH = "v3_best_old_hunter_vit.pth"

OLD_FAKE_FOLDERS = ['SD1_4', 'SD1_5', 'OpenJourney']

# ==========================================
# 2. 커스텀 Dataset 클래스 (ViT 3채널 업샘플링)
# ==========================================
class Stage1ViTDataset(Dataset):
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
        # 🌟 3채널 [Low, High, Diff] 텐서를 모두 RGB처럼 사용합니다.
        tensor = torch.load(self.filepaths[idx])
        label = self.labels[idx]
        
        for c in range(3):
            mean = tensor[c].mean()
            std = tensor[c].std()
            tensor[c] = (tensor[c] - mean) / (std + 1e-8)
            
        tensor = self.resize(tensor) # [3, 224, 224]로 부드럽게 확장
        
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
# 4. 메인 학습 파이프라인 (1단계 층화 추출)
# ==========================================
def train_and_validate():
    all_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.pt"), recursive=True)
    
    old_fake_files = [] 
    other_files = []    
    
    for fpath in all_files:
        if any(old_fake in fpath for old_fake in OLD_FAKE_FOLDERS):
            old_fake_files.append(fpath)
        else:
            other_files.append(fpath)
            
    random.seed(42)
    random.shuffle(old_fake_files)
    random.shuffle(other_files)
    
    # 8:2 층화 분할
    old_train_size = int(0.8 * len(old_fake_files))
    old_train = old_fake_files[:old_train_size]
    old_val = old_fake_files[old_train_size:]
    
    other_train_size = int(0.8 * len(other_files))
    other_train = other_files[:other_train_size]
    other_val = other_files[other_train_size:]
    
    train_files = old_train + other_train
    train_labels = [1.0] * len(old_train) + [0.0] * len(other_train)
    
    val_files = old_val + other_val
    val_labels = [1.0] * len(old_val) + [0.0] * len(other_val)

    print(f"\n📁 [Stage 1 - ViT] 완벽한 Stratified Split 분할 완료")
    print(f"   - Train Set: {len(train_files)}장 (OLD FAKE {len(old_train)} vs OTHERS {len(other_train)})")
    print(f"   - Val Set: {len(val_files)}장 (OLD FAKE {len(old_val)} vs OTHERS {len(other_val)})")

    train_dataset = Stage1ViTDataset(train_files, train_labels, is_train=True)
    val_dataset = Stage1ViTDataset(val_files, val_labels, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = get_model()
    # 데이터 불균형 방어용 가중치
    pos_weight = torch.tensor([len(other_train) / len(old_train)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print("\n🚀 [Step 1 - ViT] Old Fake Hunter 학습 시작")
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
    plt.title('Stage 1 (ViT) Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green')
    plt.title('Stage 1 (ViT) Accuracy')
    plt.legend()
    plt.savefig('stage1_vit_training_history_overfitting_solve.png', dpi=150)
    print(f"\n✅ 1단계 ViT 완료! 최고 정확도: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    train_and_validate()