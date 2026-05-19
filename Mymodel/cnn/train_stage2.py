import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. 하이퍼파라미터 및 경로 설정 ---
DATASET_DIR = "/data1/DeepFake/dataset_cnn"  # 전처리된 텐서 폴더
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL_PATH = "best_new_hunter.pth"

# 2단계 타겟
NEW_FAKE_FOLDERS = ['flux_dev', 'nano_banana', 'std_3.5_large_turbo']
OLD_FAKE_FOLDERS = ['SD1_4', 'SD1_5', 'OpenJourney']

# --- 2. 커스텀 Dataset 클래스 (1:1 밸런싱 적용) ---
class Stage2Dataset(Dataset):
    def __init__(self, root_dir):
        self.filepaths = []
        self.labels = []
        
        all_files = glob.glob(os.path.join(root_dir, "**", "*.pt"), recursive=True)
        
        real_files = []
        new_fake_files = []
        
        # 파일 분류
        for fpath in all_files:
            # 구형 FAKE는 2단계에서 완전히 버립니다.
            if any(old_fake in fpath for old_fake in OLD_FAKE_FOLDERS):
                continue
            
            if any(new_fake in fpath for new_fake in NEW_FAKE_FOLDERS):
                new_fake_files.append(fpath)
            elif "real" in fpath.lower():
                real_files.append(fpath)
                
        # 🌟 핵심: 완벽한 1:1 비율을 위해 랜덤 샘플링 (클래스 불균형 해결)
        target_count = len(new_fake_files)
        random.seed(42) # 재현성을 위해 시드 고정
        balanced_real_files = random.sample(real_files, target_count)
        
        # 데이터셋 병합
        self.filepaths.extend(new_fake_files)
        self.labels.extend([1.0] * target_count)
        
        self.filepaths.extend(balanced_real_files)
        self.labels.extend([0.0] * target_count)
        
        print(f"📊 2단계 데이터 로드 완료: 완벽한 1:1 밸런스 매치!")
        print(f"   - 타겟 [NEW FAKE (Label 1)]: {target_count}장")
        print(f"   - 비교군 [REAL (Label 0)]: {target_count}장 (16k 중 샘플링)")
        print(f"   - 총 학습 데이터: {len(self.filepaths)}장")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # [2, 64, 64] 텐서 로드 (0:High, 1:Low)
        tensor = torch.load(self.filepaths[idx])
        label = self.labels[idx]
        
        # 🌟 2단계 핵심: 신형 모델의 약점인 Low 대역(인덱스 1) 추출!
        low_err_map = tensor[1:2, :, :] 
        
        # 정규화 (Instance Norm)
        mean = low_err_map.mean()
        std = low_err_map.std()
        low_err_map = (low_err_map - mean) / (std + 1e-8)
        
        return low_err_map, torch.tensor([label], dtype=torch.float32)

# --- 3. 모델 정의 (1-Channel ResNet-18) ---
def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    
    return model.to(DEVICE)

# --- 4. 메인 학습 파이프라인 ---
def train_and_validate():
    full_dataset = Stage2Dataset(DATASET_DIR)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"📁 분할 완료: Train {train_size}장 / Validation {val_size}장")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = get_model()
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print("\n🚀 [Step 2: New Fake Hunter CNN 학습 시작]")
    for epoch in range(EPOCHS):
        # === Training ===
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
            
        avg_train_loss = train_loss / train_size
        scheduler.step()
        
        # === Validation ===
        model.eval()
        val_loss = 0.0
        corrects = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                preds = torch.sigmoid(outputs) >= 0.5
                corrects += (preds == labels).sum().item()
                
        avg_val_loss = val_loss / val_size
        val_acc = corrects / val_size
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"📈 Epoch {epoch+1} 결과: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"  🌟 Best Model 저장됨! (Val Acc: {best_val_acc*100:.2f}%)")

    # === 결과 시각화 ===
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.savefig('stage2_training_history.png', dpi=150)
    print("\n✅ 학습 완료! 'stage2_training_history.png' 차트가 저장되었습니다.")
    print(f"최종 2단계 Best Validation Accuracy: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    train_and_validate()