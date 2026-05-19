import os
import shutil
import random
from tqdm import tqdm

# =========================================================
# [설정] 경로를 본인 환경에 맞게 수정하세요!
# =========================================================

# 1. 원본 PNG들이 있는 최상위 폴더 (예: /data1/REAL)
SRC_PNG_ROOT = "/data1/Deepfake/REAL" 

# 2. 변환된 JPG들이 있는 최상위 폴더 (예: /data1/REAL_JPG)
SRC_JPG_ROOT = "/data1/Deepfake/REAL_JPG"

# 3. 섞여서 저장될 폴더 (예: /data1/REAL_MIXED)
DST_MIXED_ROOT = "/data1/Deepfake/REAL_MIXED"

# =========================================================

def create_mixed_dataset():
    print(f"🚀 REAL_MIXED 데이터셋 생성을 시작합니다!")
    print(f"🔹 PNG 소스: {SRC_PNG_ROOT}")
    print(f"🔸 JPG 소스: {SRC_JPG_ROOT}")
    print(f"🏁 결과 위치: {DST_MIXED_ROOT}\n")

    # 1. REAL 폴더 안에 있는 데이터셋 목록 가져오기 (DIV2K, Flickr2K 등)
    if not os.path.exists(SRC_PNG_ROOT):
        print(f"❌ 오류: 원본 폴더({SRC_PNG_ROOT})가 없습니다.")
        return

    datasets = [d for d in os.listdir(SRC_PNG_ROOT) if os.path.isdir(os.path.join(SRC_PNG_ROOT, d))]
    
    if not datasets:
        print("⚠️ 처리할 데이터셋 폴더가 없습니다.")
        return

    total_copied = 0

    for dataset_name in datasets:
        png_dir = os.path.join(SRC_PNG_ROOT, dataset_name)
        jpg_dir = os.path.join(SRC_JPG_ROOT, dataset_name)
        mixed_dir = os.path.join(DST_MIXED_ROOT, dataset_name)

        # JPG 폴더가 짝이 맞는지 확인
        if not os.path.exists(jpg_dir):
            print(f"⚠️ 경고: '{dataset_name}'의 JPG 폴더가 없어서 건너뜁니다.")
            continue

        # PNG 파일 목록 가져오기 (기준점)
        files = [f for f in os.listdir(png_dir) if f.lower().endswith('.png')]
        files.sort() # 순서를 고정해서 섞기 위해 정렬

        if not files:
            continue

        print(f"📦 [{dataset_name}] 처리 중... (총 {len(files)}장)")
        os.makedirs(mixed_dir, exist_ok=True)

        # 2. 50:50 섞기 (홀수: PNG, 짝수: JPG)
        for idx, filename in enumerate(tqdm(files, desc=f"Mixing {dataset_name}")):
            name_no_ext = os.path.splitext(filename)[0]
            
            # 대상 경로 설정
            target_path = ""
            
            # [전략] 인덱스가 짝수면 PNG, 홀수면 JPG (50:50)
            if idx % 2 == 0:
                # PNG 가져오기
                src_file = os.path.join(png_dir, filename)
                dst_filename = filename # .png
                
            else:
                # JPG 가져오기 (이름은 같고 확장자만 .jpg인 파일 찾기)
                src_file = os.path.join(jpg_dir, f"{name_no_ext}.jpg")
                dst_filename = f"{name_no_ext}.jpg"
                
                # 만약 해당 JPG가 없으면? -> 그냥 PNG로 대체 (비상대책)
                if not os.path.exists(src_file):
                    src_file = os.path.join(png_dir, filename)
                    dst_filename = filename

            # 최종 복사
            dst_path = os.path.join(mixed_dir, dst_filename)
            
            # 이미 있으면 건너뛰기
            if not os.path.exists(dst_path):
                shutil.copy2(src_file, dst_path)
                total_copied += 1

    print("\n" + "="*40)
    print("🎉 REAL_MIXED 생성 완료!")
    print(f"✅ 총 {total_copied}장의 파일이 50:50으로 섞여 저장되었습니다.")
    print(f"📂 확인: {DST_MIXED_ROOT}")

if __name__ == "__main__":
    create_mixed_dataset()