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

# --- 1. 하이퍼파라미터 및 경로 설정 ---
DATASET_DIR = "/data1/DeepFake/dataset_cnn"  # 전처리된 텐서 폴더
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3 # 과적합 방지를 위한 강력한 L2 정규화
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL_PATH = "best_old_hunter_v2.pth"

# 우리가 1단계에서 걸러낼 '구형 FAKE' 폴더명
OLD_FAKE_FOLDERS = ['SD1_4', 'SD1_5', 'OpenJourney']

# --- 2. 커스텀 Dataset 클래스 (외부 리스트 주입형) ---
class Stage1ExactDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # .pt 텐서 파일 로드: [3, 64, 64] (0:Low, 1:High, 2:Diff)
        tensor = torch.load(self.filepaths[idx])
        label = self.labels[idx]
        
        # 🌟 핵심: 1단계이므로 Channel 1 (High: 150-200)만 추출! -> [1, 64, 64]
        high_err_map = tensor[1:2, :, :] 
        
        # 정규화 (Instance Norm)
        mean = high_err_map.mean()
        std = high_err_map.std()
        high_err_map = (high_err_map - mean) / (std + 1e-8)
        
        # Train 시에만 전달되는 Augmentation 적용
        if self.transform:
            high_err_map = self.transform(high_err_map)
            
        return high_err_map, torch.tensor([label], dtype=torch.float32)

# --- 3. 모델 정의 (1-Channel ResNet-18 완벽 개조) ---
def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 🌟 개조 1: 1채널 입력, 3x3 필터, Stride 1 (64x64 작은 이미지 해상도 보존)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 🌟 개조 2: 해상도를 뭉개는 첫 번째 MaxPool 삭제
    model.maxpool = nn.Identity()
    
    # 🌟 개조 3: 과적합 방지용 Dropout(50%) 추가
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 1)
    )
    
    return model.to(DEVICE)

# --- 4. 메인 학습 파이프라인 ---
def train_and_validate():
    # --- [데이터 수집 및 분류] ---
    all_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.pt"), recursive=True)
    
    old_fake_files = [] # 타겟 (Label 1)
    other_files = []    # 나머지 REAL + NEW FAKE (Label 0)
    
    for fpath in all_files:
        if any(old_fake in fpath for old_fake in OLD_FAKE_FOLDERS):
            old_fake_files.append(fpath)
        else:
            other_files.append(fpath)
            
    # 리스트 섞기 (시드 고정)
    random.seed(42)
    random.shuffle(old_fake_files)
    random.shuffle(other_files)
    
    # --- 🌟 [작성자님 요청: 완벽한 층화 추출 (Stratified Split)] ---
    # OLD FAKE를 8:2로 분할
    old_train_size = int(0.8 * len(old_fake_files))
    old_train = old_fake_files[:old_train_size]
    old_val = old_fake_files[old_train_size:]
    
    # OTHERS (REAL + NEW FAKE)를 8:2로 분할
    other_train_size = int(0.8 * len(other_files))
    other_train = other_files[:other_train_size]
    other_val = other_files[other_train_size:]
    
    # Train 세트 병합 (비율 유지됨)
    train_files = old_train + other_train
    train_labels = [1.0] * len(old_train) + [0.0] * len(other_train)
    
    # Val 세트 병합 (비율 유지됨)
    val_files = old_val + other_val
    val_labels = [1.0] * len(old_val) + [0.0] * len(other_val)

    print(f"\n📁 [1단계 완벽한 Stratified Split 분할 완료]")
    print(f"   - Train Set: 총 {len(train_files)}장 (OLD FAKE {len(old_train)}장 vs OTHERS {len(other_train)}장)")
    print(f"   - Val Set: 총 {len(val_files)}장 (OLD FAKE {len(old_val)}장 vs OTHERS {len(other_val)}장)")

    # --- [Dataset 및 DataLoader 생성] ---
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
    ])
    
    train_dataset = Stage1ExactDataset(train_files, train_labels, transform=train_transform)
    val_dataset = Stage1ExactDataset(val_files, val_labels, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = get_model()
    # 양성(Label 1) 데이터가 3배 적으므로, 손실 함수에 가중치(pos_weight)를 주어 밸런스를 강제로 맞춥니다.
    pos_weight = torch.tensor([len(other_train) / len(old_train)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print("\n🚀 [Step 1 - V2: Old Fake Hunter CNN 학습 시작 (층화추출 & 수술적용)]")
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
    
    plt.savefig('stage1_v2_training_history.png', dpi=150)
    print("\n✅ 학습 완료! 'stage1_v2_training_history.png' 차트가 저장되었습니다.")
    print(f"최종 1단계 V2 Best Validation Accuracy: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    train_and_validate()