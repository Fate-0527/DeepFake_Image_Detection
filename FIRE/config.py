# config.py
import os

class Config:

    SAVE_DIR = "/data1/checkpoints/FIRE"
    # 1. 데이터 경로 설정 (사용자 환경에 맞게 수정 필요)
    REAL_DIRS = {
        # REAL Train (0 ~ 456)
        "/data1/DeepFake/REAL/CLIC": (0, 456),
        "/data1/DeepFake/REAL/DIV2K": (0, 456),
        "/data1/DeepFake/REAL/Flickr2K": (0, 456),
        "/data1/DeepFake/REAL/LSDIR": (0, 456),
        "/data1/DeepFake/REAL/RAISE": (0, 456),
        "/data1/DeepFake/REAL/UCID1338": (0, 456)
    }

    FAKE_DIRS = {
        # FAKE Train (0 ~ 456)
        "/data1/DeepFake/FAKE/flux_dev": (0, 456),
        "/data1/DeepFake/FAKE/nano_banana": (0, 456),
        "/data1/DeepFake/FAKE/std_3.5_large_turbo": (0, 456),
        "/data1/DeepFake/FAKE/OpenJourney": (0, 456),
        "/data1/DeepFake/FAKE/SD1_4": (0, 456),
        "/data1/DeepFake/FAKE/SD1_5": (0, 456)
    }

    Vaild_REAL_DIRS = {
        # VALID REAL (456 ~ 570)
        "/data1/DeepFake/REAL/CLIC": (456, 570),
        "/data1/DeepFake/REAL/DIV2K": (456, 570),
        "/data1/DeepFake/REAL/Flickr2K": (456, 570),
        "/data1/DeepFake/REAL/LSDIR": (456, 570),
        "/data1/DeepFake/REAL/RAISE": (456, 570),
        "/data1/DeepFake/REAL/UCID1338": (456, 570),
        
        # train2017은 데이터가 많으므로 비율(1:6)에 맞춰서 설정
        # (Train이 456장이면, Valid는 114장이어야 함. 
        #  전체 Real Valid 합이 684장이 되어야 하므로 train2017 없이 위 6개만으로도 충분할 수 있음.
        #  만약 train2017을 꼭 써야 한다면 구간을 지정해주세요. 일단은 주석 처리하거나 0~684로 둡니다.)
        # "/data1/DeepFake/train2017" : (0, 684) 
    }

    Vaild_FAKE_DIRS = {
        # VALID FAKE (456 ~ 570)
        "/data1/DeepFake/FAKE/flux_dev": (456, 570),
        "/data1/DeepFake/FAKE/nano_banana": (456, 570),
        "/data1/DeepFake/FAKE/std_3.5_large_turbo": (456, 570),
        "/data1/DeepFake/FAKE/OpenJourney": (456, 570),
        "/data1/DeepFake/FAKE/SD1_4": (456, 570),
        "/data1/DeepFake/FAKE/SD1_5": (456, 570),
        
        # VALID FAKE JPG (570 ~ 684)
        "/data1/DeepFake/FAKE_JPG/flux_dev": (456, 570),
        "/data1/DeepFake/FAKE_JPG/nano_banana": (570, 684),
        "/data1/DeepFake/FAKE_JPG/std_3.5_large_turbo": (570, 684),
        "/data1/DeepFake/FAKE_JPG/OpenJourney": (570, 684),
        "/data1/DeepFake/FAKE_JPG/SD1_4": (570, 684),
        "/data1/DeepFake/FAKE_JPG/SD1_5": (570, 684)
    }

    
    # 3. Master-Replica 설정 (On/Off 스위치)
    USE_MASTER_REPLICA = False      # 초기 성능 확인 시 False, 강건성 테스트 시 True
    COMPRESSION_PROB = 0.5         # 훈련 중 JPEG 압축을 적용할 확률
    JPEG_QUALITY_RANGE = (30, 100) # JPEG 압축 품질 범위

    # 4. 하이퍼파라미터
    IMG_SIZE = 256                 # 입력 이미지 크기
    BATCH_SIZE = 16                # GPU 메모리에 따라 조절 (8, 16, 32 등)
    NUM_EPOCHS = 10                # 훈련 에포크 수
    LEARNING_RATE = 1e-4           # 분류기 학습률
    NUM_WORKERS = 4                # 데이터 로딩에 사용할 CPU 프로세스 수
    RANDOM_SEED = 42               # 재현성을 위한 랜덤 시드

        # 5. 저장 및 기타
    SAVE_DIR = "/data1/checkpoints/FIRE"      # 모델 체크포인트 저장 경로
    RESULT_DIR = "./result_png"        # 결과 PNG 이미지 저장 경로
    DETAIL_FILE = "detail_analysis_2"   # 상세 분석 결과 저장 경로
    PNG_FILE = "fire2.png"  # 결과 PNG 파일 이름
    RANDOM_SEED = 42               # 재현성을 위한 시드값
    FILE_NAME = "fire2_bin"        # 저장되는 모델 파일 이름 접두사


# 저장 경로 자동 생성
os.makedirs(Config.SAVE_DIR, exist_ok=True)