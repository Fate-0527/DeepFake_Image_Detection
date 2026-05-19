import os
import glob
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# 사용자님의 설정 파일
from config import Config

def collect_paths_from_config(dir_dict):
    """Config.py의 딕셔너리({경로: 개수})를 읽어 파일 리스트 생성"""
    all_paths = []
    for dir_path, count in dir_dict.items():
        # jpg, png, jpeg 등 대소문자 구분 없이 검색
        patterns = ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(dir_path, "**", p), recursive=True))
        
        # 파일이 모자라면 전체 사용, 남으면 랜덤 샘플링
        if len(files) > count:
            files = random.sample(files, count)
        
        all_paths.extend(files)
    return all_paths

class CustomFIREDataset(Dataset):
    """FIRE 모델 학습을 위한 데이터셋 (Config 연동 버전)"""
    def __init__(self, real_paths, fake_paths, is_train=True):
        self.image_paths = real_paths + fake_paths
        # Real=0, Fake=1
        self.labels = [0] * len(real_paths) + [1] * len(fake_paths)
        
        # 공식 코드와 동일한 Transform 적용 (Resize -> ToTensor)
        # FIRE 논문은 학습 시 별도의 강한 증강을 기본 코드에선 안 쓰는 경향이 있어 기본만 유지하되,
        # 원하신다면 여기에 Augmentation 추가 가능합니다.
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            # FIRE 공식 코드는 Normalize를 주석 처리해두었으므로 여기서도 뺍니다.
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            label = torch.tensor(self.labels[idx], dtype=torch.long) # 정수형 라벨
            return image, label
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            # 에러 시 다음 이미지 리턴 (임시 방편)
            return self.__getitem__((idx + 1) % len(self))