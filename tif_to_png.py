import os
import glob
from PIL import Image
from tqdm import tqdm

# =========================================================
# [설정] 경로를 맞춰주세요!
# =========================================================

# 1. TIF 파일들이 들어있는 폴더 (이 안을 다 뒤집니다)
SOURCE_DIR = "/data1/Deepfake/REAL/UCID1338" 

# 2. PNG로 변환되어 저장될 폴더
SAVE_DIR = "/data1/Deepfake/REAL/UCID1338_PNG"

# =========================================================

# 대용량 이미지 처리 시 경고 끄기 (Pillow 안전장치 해제)
Image.MAX_IMAGE_PIXELS = None

def convert_tif_to_png():
    # 1. 모든 .tif / .tiff 파일 찾기
    print(f"🔍 {SOURCE_DIR} 에서 TIF 파일 검색 중...")
    # 확장자 대소문자 구분 없이 찾기 위해 패턴 2개 사용
    tif_files = glob.glob(os.path.join(SOURCE_DIR, "**", "*.tif"), recursive=True)
    tiff_files = glob.glob(os.path.join(SOURCE_DIR, "**", "*.tiff"), recursive=True)
    all_files = tif_files + tiff_files
    
    if not all_files:
        print(f"❌ '{SOURCE_DIR}' 안에 .tif 파일이 없습니다.")
        return

    print(f"📦 총 {len(all_files)}개의 파일을 변환합니다.")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 2. 변환 시작
    success_count = 0
    
    for idx, file_path in enumerate(tqdm(all_files, desc="변환 중")):
        try:
            # 원본 파일명 가져오기
            filename = os.path.basename(file_path)
            name_only, _ = os.path.splitext(filename)
            
            # 저장할 경로 설정 (파일명.png)
            save_path = os.path.join(SAVE_DIR, f"{name_only}.png")
            
            # 이미 변환된 파일이 있으면 건너뛰기 (시간 절약)
            if os.path.exists(save_path):
                continue

            with Image.open(file_path) as img:
                # RGB 모드로 변환 (TIFF는 CMYK나 16bit인 경우가 많아 호환성을 위해 변환)
                img = img.convert("RGB")
                img.save(save_path, "PNG")
                success_count += 1
                
        except Exception as e:
            print(f"\n⚠️ 변환 실패 ({os.path.basename(file_path)}): {e}")

    print("\n" + "="*40)
    print(f"🎉 변환 완료!")
    print(f"성공: {success_count} / {len(all_files)}")
    print(f"저장 위치: {SAVE_DIR}")

if __name__ == "__main__":
    convert_tif_to_png()