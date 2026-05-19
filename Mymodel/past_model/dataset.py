import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import jpeg_compress

# ★ 새로 추가된 핵심 함수 ★
def collect_image_paths(dir_config):
    """
    폴더별로 지정된 개수만큼 이미지를 랜덤 샘플링하여 수집
    Args:
        dir_config: dict ({"폴더경로": 개수, ...})
                    개수가 None이면 해당 폴더의 모든 이미지 사용
    Returns:
        list of str (수집된 모든 이미지 경로)
    """
    all_paths = []
    
    # 리스트가 들어오면 딕셔너리로 변환 (호환성)
    if isinstance(dir_config, list):
        dir_config = {d: None for d in dir_config}

    for directory, limit in dir_config.items():
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            continue
            
        current_folder_paths = []
        for fname in os.listdir(directory):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')):
                current_folder_paths.append(os.path.join(directory, fname))
        
        # 1. 셔플 (랜덤 뽑기)
        random.shuffle(current_folder_paths)
        
        # 2. 개수 자르기 (이게 핵심!)
        total_files = len(current_folder_paths)
        if limit is not None:
            if limit > total_files:
                limit = total_files
            print(f"[Sampling] {directory}: {total_files}장 중 {limit}장 선택")
            current_folder_paths = current_folder_paths[:limit]
        else:
            print(f"[All] {directory}: 전체 {total_files}장 사용")
            
        all_paths.extend(current_folder_paths)
        
    return all_paths

def split_dataset(real_paths, fake_paths, train_ratio=0.8, max_samples=None, seed=42):
    """
    Real/Fake 데이터를 train/test로 분할 (최대 개수 제한 적용)
    """
    random.seed(seed)
    
    # 1. 셔플 (랜덤 섞기)
    real_shuffled = real_paths.copy()
    fake_shuffled = fake_paths.copy()
    random.shuffle(real_shuffled)
    random.shuffle(fake_shuffled)
    
    # 2. 개수 제한 적용 (밸런싱)
    if max_samples is not None:
        # Real 데이터가 max_samples보다 적으면 전체 다 씀
        n_real = min(len(real_shuffled), max_samples)
        # Fake 데이터도 max_samples만큼만 자름 (18000개 → 9000개)
        n_fake = min(len(fake_shuffled), max_samples)
        
        real_shuffled = real_shuffled[:n_real]
        fake_shuffled = fake_shuffled[:n_fake]
        
        print(f"Sampling applied: Real={n_real}, Fake={n_fake} (Limit={max_samples})")
    
    # 3. Train/Test 분할
    # Real 분할
    split_idx_real = int(len(real_shuffled) * train_ratio)
    train_real = real_shuffled[:split_idx_real]
    test_real = real_shuffled[split_idx_real:]
    
    # Fake 분할
    split_idx_fake = int(len(fake_shuffled) * train_ratio)
    train_fake = fake_shuffled[:split_idx_fake]
    test_fake = fake_shuffled[split_idx_fake:]
    
    print(f"Dataset Split:")
    print(f"  Train: Real={len(train_real)}, Fake={len(train_fake)}")
    print(f"  Test:  Real={len(test_real)}, Fake={len(test_fake)}")
    
    return train_real, train_fake, test_real, test_fake


class MasterReplicaDataset(Dataset):
    """
    Master-Replica 파이프라인 데이터셋
    """
    def __init__(self, real_paths, fake_paths, img_size=224, 
                 use_master_replica=True, compression_prob=0.5,
                 jpeg_quality_range=(30, 100), mode='train'):
      
        self.real_paths = real_paths
        self.fake_paths = fake_paths
        
        self.paths = self.real_paths + self.fake_paths
        self.labels = [0] * len(self.real_paths) + [1] * len(self.fake_paths)
        
        self.use_master_replica = use_master_replica
        self.compression_prob = compression_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.mode = mode
        
        # 기본 전처리
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]
        
        # PNG(Master) 로드
        image = Image.open(img_path).convert('RGB')
        
        # Master-Replica 파이프라인 적용 (학습 시에만)
        if self.mode == 'train' and self.use_master_replica:
            if random.random() < self.compression_prob:
                # Replica: 랜덤 JPEG 압축 적용
                quality = random.randint(*self.jpeg_quality_range)
                image = jpeg_compress(image, quality)
            # else: Master(PNG) 그대로 사용
        
        # 전처리
        image = self.transform(image)
        
        return image, label


class EvaluationDataset(Dataset):
    """
    평가용 데이터셋 (특정 압축 레벨 적용)
    """
    def __init__(self, real_paths, fake_paths, img_size=224, compression_type='png'):
        """
        Args:
            real_paths: list (Real 이미지 경로)
            fake_paths: list (Fake 이미지 경로)
            compression_type: 'png', 'jpg_90', 'jpg_70', 'jpg_50', 'jpg_30' 등
        """
        from utils import apply_compression
        
        self.real_paths = real_paths
        self.fake_paths = fake_paths
        
        self.paths = self.real_paths + self.fake_paths
        self.labels = [0] * len(self.real_paths) + [1] * len(self.fake_paths)
        self.compression_type = compression_type
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size), # 짧은 변을 img_size에 맞춤 (비율 유지)
            transforms.RandomCrop(img_size), # 랜덤하게 512x512 영역 뜯어내기
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        from utils import apply_compression
        
        img_path = self.paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = apply_compression(image, self.compression_type)
        image = self.transform(image)
        
        return image, label
