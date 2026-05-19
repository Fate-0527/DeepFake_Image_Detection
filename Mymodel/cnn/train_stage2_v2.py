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
DATASET_DIR = "/data1/DeepFake/dataset_cnn"  # 전처리된 텐서 폴더
BATCH_SIZE = 128
EPOCHS = 50 # Augmentation(증강)이 들어갔으므로 에포크를 조금 늘려줍니다.
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3 # 과적합 방지를 위한 강력한 L2 정규화
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL_PATH = "best_new_hunter_v2.pth"

NEW_FAKE_FOLDERS = ['flux_dev', 'nano_banana', 'std_3.5_large_turbo']
OLD_FAKE_FOLDERS = ['SD1_4', 'SD1_5', 'OpenJourney']

# ==========================================
# 2. 커스텀 Dataset 클래스 (외부 리스트 주입형)
# ==========================================
class ExactBalancedDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # 텐서 로드: 원래는 [3, 64, 64] (0:Low, 1:High, 2:Diff)
        tensor = torch.load(self.filepaths[idx])
        label = self.labels[idx]
        
        # 🌟 2채널(Low, High) 컷팅! 3번째 Diff 채널은 버립니다.
        tensor = tensor[:2, :, :] 
        
        # 채널별 독립적 정규화 (Instance Norm)
        for c in range(2):
            mean = tensor[c].mean()
            std = tensor[c].std()
            tensor[c] = (tensor[c] - mean) / (std + 1e-8)
        
        # Train 시에만 전달되는 Augmentation 적용
        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor, torch.tensor([label], dtype=torch.float32)

# ==========================================
# 3. 모델 정의 (ResNet-18 수술 완료)
# ==========================================
def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 🌟 개조 1: 2채널 입력, 3x3 필터, Stride 1 (작은 이미지 해상도 보존)
    model.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 🌟 개조 2: 해상도를 반토막 내는 첫 번째 MaxPool 삭제 (정보 보존)
    model.maxpool = nn.Identity()
    
    # 🌟 개조 3: 출력단에 강력한 Dropout(50%) 추가하여 과적합 억제
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 1)
    )
    
    return model.to(DEVICE)

# ==========================================
# 4. 메인 학습 파이프라인
# ==========================================
def train_and_validate():
    # --- [데이터 수집 및 층화 추출 분할] ---
    all_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.pt"), recursive=True)
    real_files = []
    new_fake_files = []
    
    for fpath in all_files:
        if any(old_fake in fpath for old_fake in OLD_FAKE_FOLDERS): 
            continue # 구형 모델은 학습에서 완전히 배제
        
        if any(new_fake in fpath for new_fake in NEW_FAKE_FOLDERS):
            new_fake_files.append(fpath)
        elif "real" in fpath.lower():
            real_files.append(fpath)
            
    # 전체 1:1 밸런싱 (신형 FAKE 개수에 맞춰 REAL을 랜덤 샘플링)
    target_count = len(new_fake_files)
    random.seed(42)
    balanced_real_files = random.sample(real_files, target_count)
    
    # 리스트 섞기
    random.shuffle(balanced_real_files)
    random.shuffle(new_fake_files)
    
    # 🌟 작성자님 아이디어: 층화 추출 (Stratified Split) - 각각 8:2로 분할
    train_size = int(0.8 * target_count)
    
    # Train 구성 (REAL 80% + FAKE 80% 합치기) = 완벽한 1:1 보장
    train_files = balanced_real_files[:train_size] + new_fake_files[:train_size]
    train_labels = [0.0] * train_size + [1.0] * train_size
    
    # Val 구성 (REAL 20% + FAKE 20% 합치기) = 완벽한 1:1 보장
    val_files = balanced_real_files[train_size:] + new_fake_files[train_size:]
    val_labels = [0.0] * (target_count - train_size) + [1.0] * (target_count - train_size)

    print(f"\n📁 [완벽한 Stratified Split 분할 완료]")
    print(f"   - Train Set: 총 {len(train_files)}장 (REAL {train_size}장 vs FAKE {train_size}장)")
    print(f"   - Val Set: 총 {len(val_files)}장 (REAL {target_count - train_size}장 vs FAKE {target_count - train_size}장)")

    # --- [Dataset 및 DataLoader 생성] ---
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
    ])
    
    # Train에만 증강 적용, Val은 원본 그대로 평가
    train_dataset = ExactBalancedDataset(train_files, train_labels, transform=train_transform)
    val_dataset = ExactBalancedDataset(val_files, val_labels, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # --- [모델, 손실함수, 옵티마이저 세팅] ---
    model = get_model()
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print("\n🚀 [Step 2 - V2: 2-Channel 퓨전 & 완벽 균형 학습 시작]")
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
            
        avg_train_loss = train_loss / len(train_dataset)
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
                
        avg_val_loss = val_loss / len(val_dataset)
        val_acc = corrects / len(val_dataset)
        
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
    plt.title('Loss History V2')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green')
    plt.title('Validation Accuracy V2')
    plt.legend()
    
    plt.savefig('stage2_v2_training_history.png', dpi=150)
    print("\n✅ 학습 완료! 'stage2_v2_training_history.png' 차트가 저장되었습니다.")
    print(f"최종 2단계 V2 Best Validation Accuracy: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    train_and_validate()