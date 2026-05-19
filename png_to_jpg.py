import os
from PIL import Image
from tqdm import tqdm

# =========================================================
# [설정] 경로를 맞춰주세요!
# =========================================================

# 1. PNG 파일들이 있는 곳 (입력)
SOURCE_ROOT = "/data1/DeepFake/FAKE"

# 2. JPG 파일들을 저장할 곳 (출력)
# (원본 폴더 구조를 그대로 따라갑니다)
DEST_ROOT = "/data1/DeepFake/FAKE_JPG"
# 3. JPG 저장 품질 (1~100, 보통 95 권장)
JPG_QUALITY = 95

# =========================================================

print(f"🚀 PNG -> JPG 변환 시작!")
print(f"📂 입력: {SOURCE_ROOT}")
print(f"📂 출력: {DEST_ROOT}")

# 파일 목록 수집
all_files = []
for root, _, files in os.walk(SOURCE_ROOT):
    for file in files:
        if file.lower().endswith('.png'):
            all_files.append(os.path.join(root, file))

print(f"📦 총 {len(all_files)}개의 PNG 파일을 발견했습니다.")

if not all_files:
    print("❌ 변환할 파일이 없습니다.")
    exit()

# 변환 시작
success_count = 0
error_count = 0

for src_path in tqdm(all_files, desc="변환 중"):
    try:
        # 저장할 경로 계산 (폴더 구조 유지)
        # 예: /data1/CLIC/train/image.png -> /data1/CLIC_JPG/train/image.jpg
        relative_path = os.path.relpath(src_path, SOURCE_ROOT)
        name_no_ext = os.path.splitext(relative_path)[0]
        dest_path = os.path.join(DEST_ROOT, f"{name_no_ext}.jpg")
        
        # 폴더가 없으면 생성
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # 이미 변환된 파일 있으면 건너뛰기 (이어하기 기능)
        if os.path.exists(dest_path):
            continue

        with Image.open(src_path) as img:
            # [중요] PNG(RGBA)를 JPG(RGB)로 변환
            # 투명 채널이 있으면 에러나므로 RGB로 변환 필수
            rgb_im = img.convert('RGB')
            
            # JPG로 저장
            rgb_im.save(dest_path, quality=JPG_QUALITY)
            success_count += 1
            
    except Exception as e:
        # print(f"에러: {src_path} - {e}") # 에러 로그 필요 시 주석 해제
        error_count += 1

print("\n" + "="*40)
print(f"🎉 변환 완료!")
print(f"✅ 성공: {success_count} 장")
print(f"❌ 실패: {error_count} 장")
print(f"📂 저장 위치: {DEST_ROOT}")