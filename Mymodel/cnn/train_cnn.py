import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. 하이퍼파라미터 설정 ---
DATASET_DIR = "dataset_cnn"  # 전처리 코드가 데이터를 저장한 최상위 폴더
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL_PATH = "best_error_map_resnet.pth"

# --- 2. 커스텀 Dataset 클래스 ---
class ErrorMapDataset(Dataset):
    def __init__(self, root_dir):
        self.filepaths = []
        self.labels = []
        
        # REAL 데이터 로드 (Label: 0)
        real_dir = os.path.join(root_dir, "real")
        real_files = glob.glob(os.path.join(real_dir, "**", "*.pt"), recursive=True)
        self.filepaths.extend(real_files)
        self.labels.extend([0.0] * len(real_files))
        
        # FAKE 데이터 로드 (Label: 1)
        fake_dir = os.path.join(root_dir, "fake")
        fake_files = glob.glob(os.path.join(fake_dir, "**", "*.pt"), recursive=True)
        self.filepaths.extend(fake_files)
        self.labels.extend([1.0] * len(fake_files))
        
        print(f"📊 데이터셋 로드 완료: 총 {len(self.filepaths)}장 (Real: {len(real_files)}장 / Fake: {len(fake_files)}장)")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # .pt 텐서 파일 로드: shape [3, 64, 64]
        tensor = torch.load(self.filepaths[idx])
        label = self.labels[idx]
        
        # (선택 사항) Instance Normalization: 값이 너무 작거나 클 경우를 대비해 스케일링
        # ResNet은 입력값이 정규화되어 있을 때 학습이 가장 빠름
        mean = tensor.mean(dim=[1, 2], keepdim=True)
        std = tensor.std(dim=[1, 2], keepdim=True)
        tensor = (tensor - mean) / (std + 1e-8)
        
        return tensor, torch.tensor([label], dtype=torch.float32)

# --- 3. 모델 정의 (ResNet-18) ---
def get_model():
    # Pretrained ResNet-18 로드
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 마지막 Fully Connected Layer를 이진 분류(1 output node)용으로 수정
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    
    return model.to(DEVICE)

# --- 4. 메인 학습 파이프라인 ---
def train_and_validate():
    # 데이터셋 준비 및 분할
    full_dataset = ErrorMapDataset(DATASET_DIR)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"📁 분할 완료: Train {train_size}장 / Validation {val_size}장")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 모델, 손실 함수, 옵티마이저 설정
    model = get_model()
    criterion = nn.BCEWithLogitsLoss() # 이진 분류에 최적화된 Loss (Sigmoid 내장)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print("\n🚀 [CNN 학습 시작]")
    for epoch in range(EPOCHS):
        # === Training ===
        model.train()
        train_loss = 0.0
        
        # tqdm으로 프로그레스 바 표시
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix({'Loss': loss.item()})
            
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
                
                # 예측값 계산 (Logit을 Sigmoid에 통과시켜 0.5 기준으로 판별)
                preds = torch.sigmoid(outputs) >= 0.5
                corrects += (preds == labels).sum().item()
                
        avg_val_loss = val_loss / val_size
        val_acc = corrects / val_size
        
        # 기록
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"📈 Epoch {epoch+1} 결과: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # 베스트 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"  🌟 Best Model 저장됨! (Val Acc: {best_val_acc*100:.2f}%)")

    # === 학습 결과 시각화 ===
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
    
    plt.savefig('cnn_training_history.png', dpi=150)
    print("\n✅ 학습 완료! 'cnn_training_history.png' 차트가 저장되었습니다.")
    print(f"최종 베스트 Validation Accuracy: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    train_and_validate()