import io
import random
import numpy as np
from PIL import Image
import torch
import os

def set_seed(seed=42):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def jpeg_compress(image, quality):
    """
    PIL Image를 JPEG로 압축 후 다시 PIL로 반환
    Args:
        image: PIL.Image
        quality: int (1-100)
    Returns:
        PIL.Image (압축된 이미지)
    """
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')

def apply_compression(image, compression_type):
    """
    평가 시 특정 압축 적용
    Args:
        image: PIL.Image
        compression_type: str ('png', 'jpg_90', 'jpg_50' 등)
    Returns:
        PIL.Image
    """
    if compression_type == 'png' or compression_type == 'raw':
        return image
    elif compression_type.startswith('jpg_'):
        quality = int(compression_type.split('_')[1])
        return jpeg_compress(image, quality)
    else:
        raise ValueError(f"Unknown compression type: {compression_type}")

class AverageMeter:
    """평균 및 현재 값을 추적하는 클래스"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
