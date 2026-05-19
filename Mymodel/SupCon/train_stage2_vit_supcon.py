import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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
PATIENCE = 15
LABEL_SMOOTHING = 0.05
LAMBDA_WEIGHT = 0.5 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL_PATH = "/data1/DeepFake/Frequency_Diff/v4_best_stage2_supcon_vit.pth"

NEW_FAKE_FOLDERS = ['flux_dev', 'nano_banana', 'std_3.5_large_turbo']
OLD_FAKE_FOLDERS = ['SD1_4', 'SD1_5', 'OpenJourney']

os.makedirs("/data1/DeepFake/Frequency_Diff", exist_ok=True)

# ==========================================
# 2. SupCon Loss 구현
# ==========================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = - mean_log_prob_pos
        return loss.mean()

# ==========================================
# 3. 커스텀 Dataset 클래스 
# ==========================================
class Stage2ViTSupConDataset(Dataset):
    def __init__(self, filepaths, bce_labels, supcon_labels, is_train=False):
        self.filepaths = filepaths
        self.bce_labels = bce_labels       # 0.0(Real) or 1.0(Fake)
        self.supcon_labels = supcon_labels # 0(Real) or 2(New Fake)
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
        tensor = torch.load(self.filepaths[idx])
        b_label = self.bce_labels[idx]
        s_label = self.supcon_labels[idx]
        
        for c in range(3):
            mean = tensor[c].mean()
            std = tensor[c].std()
            tensor[c] = (tensor[c] - mean) / (std + 1e-8)
            
        tensor = self.resize(tensor)
        
        if self.is_train:
            tensor = self.augment(tensor)
            b_label = b_label * (1 - LABEL_SMOOTHING) + (1 - b_label) * LABEL_SMOOTHING
            
        return tensor, torch.tensor([b_label], dtype=torch.float32), torch.tensor(s_label, dtype=torch.long)

# ==========================================
# 4. 모델 정의 (ViT SupCon)
# ==========================================
class ViTSupCon(nn.Module):
    def __init__(self):
        super(ViTSupCon, self).__init__()
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Identity() 
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.in_features, self.in_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.in_features, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        proj_features = self.projection_head(features)
        logits = self.classifier(features)
        return proj_features, logits

# ==========================================
# 5. 메인 학습 파이프라인
# ==========================================
def train_and_validate():
    all_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.pt"), recursive=True)
    real_files, new_fake_files = [], []
    
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
    
    train_size = int(0.8 * target_count)
    
    train_files = balanced_real_files[:train_size] + new_fake_files[:train_size]
    # BCE Labels: Real = 0.0, New Fake = 1.0
    train_bce_labels = [0.0] * train_size + [1.0] * train_size
    # SupCon Labels: Real = 0, New Fake = 2
    train_supcon_labels = [0] * train_size + [2] * train_size
    
    val_files = balanced_real_files[train_size:] + new_fake_files[train_size:]
    val_bce_labels = [0.0] * (target_count - train_size) + [1.0] * (target_count - train_size)
    val_supcon_labels = [0] * (target_count - train_size) + [2] * (target_count - train_size)

    print(f"\n�� [Stage 2 - SupCon] 1:1 Split 완료")
    print(f"   - Train Set: 총 {len(train_files)}장 (REAL {train_size} vs FAKE {train_size})")
    print(f"   - Val Set: 총 {len(val_files)}장 (REAL {target_count - train_size} vs FAKE {target_count - train_size})")

    train_dataset = Stage2ViTSupConDataset(train_files, train_bce_labels, train_supcon_labels, is_train=True)
    val_dataset = Stage2ViTSupConDataset(val_files, val_bce_labels, val_supcon_labels, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = ViTSupCon().to(DEVICE)
    
    criterion_bce = nn.BCEWithLogitsLoss() 
    criterion_supcon = SupConLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print("\n🚀 [Step 2 - SupCon] 학습 시작 (Real vs New Fake)")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, b_labels, s_labels in train_bar:
            inputs, b_labels, s_labels = inputs.to(DEVICE), b_labels.to(DEVICE), s_labels.to(DEVICE)
            optimizer.zero_grad()
            
            proj_features, logits = model(inputs)
            
            loss_sup = criterion_supcon(proj_features, s_labels)
            loss_b = criterion_bce(logits, b_labels)
            loss = LAMBDA_WEIGHT * loss_sup + (1 - LAMBDA_WEIGHT) * loss_b
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'SupCon': f"{loss_sup.item():.4f}", 'BCE': f"{loss_b.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_dataset)
        scheduler.step()
        
        model.eval()
        val_loss, corrects = 0.0, 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ")
        with torch.no_grad():
            for inputs, b_labels, s_labels in val_bar:
                inputs, b_labels, s_labels = inputs.to(DEVICE), b_labels.to(DEVICE), s_labels.to(DEVICE)
                proj_features, logits = model(inputs)
                
                loss_sup = criterion_supcon(proj_features, s_labels)
                loss_b = criterion_bce(logits, b_labels)
                loss = LAMBDA_WEIGHT * loss_sup + (1 - LAMBDA_WEIGHT) * loss_b
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(logits) >= 0.5
                corrects += (preds == b_labels).sum().item()
                
        avg_val_loss = val_loss / len(val_dataset)
        val_acc = corrects / len(val_dataset)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"📈 Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"  🌟 Best Model 저장됨! (Val Acc: {best_val_acc*100:.2f}%, Val Loss: {avg_val_loss:.4f})")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early Stopping! (Epoch {epoch+1})")
                break

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Stage 2 (SupCon) Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='purple')
    plt.title('Stage 2 (SupCon) Accuracy')
    plt.legend()
    plt.savefig('stage2_supcon_history.png', dpi=150)
    print(f"\n✅ 2단계 SupCon 완료! 최고 정확도: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    train_and_validate()
