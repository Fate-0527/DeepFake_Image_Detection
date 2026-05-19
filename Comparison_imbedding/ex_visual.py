# 시각화 확인용 코드 (실행해보세요)
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np
import os

SAVE_DIR = "./visualization"
os.makedirs(SAVE_DIR, exist_ok=True)

def visualize_residuals(image_path, save_name):
    if not os.path.exists(image_path): return
    
    # 1. 원본 로드
    img = Image.open(image_path).convert("L") # 흑백 변환
    
    # 2. 블러링 (내용 추출)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=3))
    
    # 3. 차이 계산 (아티팩트 추출)
    # 잘 보이게 하기 위해 차이값에 5배를 곱하고 127을 더합니다.
    diff = np.array(img, dtype=np.float32) - np.array(blurred, dtype=np.float32)
    diff = diff * 5.0 + 127.0 
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    
    # 4. 저장 및 출력
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original (Content)")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Residual (Artifacts Only)")
    plt.imshow(diff, cmap='gray')
    plt.axis('off')
    
    save_path = f"{SAVE_DIR}/vis_{save_name}.png"
    plt.savefig(save_path)
    print(f"Saved visualization: {save_path}")
    plt.close()

# 실행
visualize_residuals("/data1/DeepFake/FAKE/flux_dev/flux_dev__000000.png", "flux_noise")
visualize_residuals("/data1/DeepFake/FAKE/std_3.5_large_turbo/std_3.5_large_turbo_000000.png", "sd_noise")