import os
import glob
import random
# config.py가 업데이트 되었다고 가정하고 불러옵니다.
from config import Config

def get_image_list_by_range(dir_dict, label, type_name="Data"):
    data_list = []
    
    # 찾을 확장자 목록
    extensions = [
        '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', 
        '*.png', '*.PNG', '*.webp', '*.WEBP', 
        '*.bmp', '*.BMP', '*.tif', '*.tiff'
    ]

    for dir_path, range_tuple in dir_dict.items():
        # 1. 범위(Range) 파싱: (start, end)
        try:
            if isinstance(range_tuple, (list, tuple)) and len(range_tuple) == 2:
                start_idx = int(range_tuple[0])
                end_idx = int(range_tuple[1])
            else:
                # 튜플이 아니면 에러 처리 혹은 전체 범위로 간주
                print(f"❌ [Error] 범위 설정 오류 ({os.path.basename(dir_path)}): {range_tuple}")
                continue
        except Exception as e:
            print(f"❌ [Error] 범위 파싱 실패: {e}")
            continue

        # 2. 이미지 파일 찾기
        files = []
        for ext in extensions:
            found = glob.glob(os.path.join(dir_path, '**', ext), recursive=True)
            files.extend(found)
        
        # [매우 중요] 순서를 보장하기 위해 반드시 정렬 (파일명 순)
        files = sorted(list(set(files)))

        if not files:
            print(f"⚠️ [Skip] 이미지 없음: {os.path.basename(dir_path)}")
            continue

        # ---------------------------------------------------------
        # [핵심 로직] 지정된 인덱스 범위(start:end)로 자르기
        # ---------------------------------------------------------
        # 범위가 파일 전체 개수를 벗어나도 Python 슬라이싱은 에러 없이 가능한 만큼만 가져옵니다.
        selected = files[start_idx:end_idx]
        
        actual_count = len(selected)
        expected_count = end_idx - start_idx
        
        # 상태 메시지 생성
        if actual_count == expected_count:
            msg = f"👌 정상 ({start_idx}~{end_idx})"
        elif actual_count == 0:
            msg = f"❌ 범위 벗어남 (파일 {len(files)}개인데 {start_idx}~{end_idx} 요청)"
        else:
            msg = f"⚠️ 부족 ({start_idx}~{end_idx} 요청 -> {actual_count}장만 선택됨)"

        print(f"✅ [{type_name}] {os.path.basename(dir_path)}: 총 {len(files)}장 중 -> {msg}")

        for f in selected:
            data_list.append(f"{os.path.abspath(f)}\t{label}")

    return data_list

def main():
    print("\n--- [1] Training Data Generating (By Range) ---")
    # Config에서 (0, 456) 등으로 설정된 범위를 읽어옴
    real_train = get_image_list_by_range(Config.REAL_DIRS, 0, "Train-Real")
    fake_train = get_image_list_by_range(Config.FAKE_DIRS, 1, "Train-Fake")
    full_train = real_train + fake_train
    
    # 리스트 순서는 섞어서 저장 (훈련 시 배치 구성을 위해)
    random.shuffle(full_train)
    
    os.makedirs('annotation', exist_ok=True)
    with open('annotation/my_train_list.txt', 'w') as f:
        f.write('\n'.join(full_train))
    print(f"🎉 Train List Saved: {len(full_train)} lines -> annotation/my_train_list.txt")

    print("\n--- [2] Validation Data Generating (By Range) ---")
    # Config에서 (456, 570) 등으로 설정된 범위를 읽어옴
    real_val = get_image_list_by_range(Config.Vaild_REAL_DIRS, 0, "Valid-Real")
    fake_val = get_image_list_by_range(Config.Vaild_FAKE_DIRS, 1, "Valid-Fake")
    full_val = real_val + fake_val
    
    # Valid는 보통 섞지 않지만, 필요하다면 섞어도 무방
    # random.shuffle(full_val)
    
    with open('annotation/my_val_list.txt', 'w') as f:
        f.write('\n'.join(full_val))
    print(f"🎉 Valid List Saved: {len(full_val)} lines -> annotation/my_val_list.txt")

if __name__ == "__main__":
    main()