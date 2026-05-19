import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import os
import random
import numpy as np

from utils.network_utils import FIRE_model
from fire_dataset_adapter import CustomFIREDataset, collect_paths_from_config
from config import Config

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 데이터셋 준비 (Test split 재현)
    real_paths = collect_paths_from_config(Config.REAL_DIRS)
    fake_paths = collect_paths_from_config(Config.FAKE_DIRS)
    
    # 학습 때와 동일한 시드로 셔플해야 Test셋이 오염 안 됨
    random.seed(Config.RANDOM_SEED) 
    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    
    train_r = int(len(real_paths) * Config.TRAIN_RATIO)
    train_f = int(len(fake_paths) * Config.TRAIN_RATIO)
    
    test_real = real_paths[train_r:]
    test_fake = fake_paths[train_f:]
    
    print(f">>> Test Set: {len(test_real)} Real, {len(test_fake)} Fake")
    
    test_ds = CustomFIREDataset(test_real, test_fake, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. 모델 로드
    model = FIRE_model(mode="frq").to(device)
    ckpt_path = os.path.join(Config.SAVE_DIR, "FIRE_Official_Replication", "best_fire_model.pth")
    
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print("✅ Official FIRE Model Loaded!")
    else:
        print("❌ Model not found. Train first.")
        return

    # 3. 평가
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            
            # FIRE Forward
            out, _, _, _, _ = model(imgs)
            
            # Sigmoid로 확률 변환 (out[:, 0]이 로짓임)
            probs = torch.sigmoid(out[:, 0])
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # 4. 결과 출력
    acc = accuracy_score(all_labels, np.array(all_probs) > 0.5)
    auc = roc_auc_score(all_labels, all_probs)
    
    print("-" * 30)
    print(f"🔥 Official FIRE Model Results")
    print(f"   Accuracy : {acc*100:.2f}%")
    print(f"   AUROC    : {auc*100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    main()