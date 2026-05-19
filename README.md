# DeepFake Image Detection Repository

본 저장소는 딥페이크 이미지 탐지(Detection)에 관한 연구 모델 및 검증 프레임워크들을 포함하고 있는 리포지토리입니다.

## 📂 디렉터리 구조 및 설명 (Directory Structure)

1. **`Mymodel/` (핵심 연구 모델)**
   - 사용자가 직접 개발/커스텀 중인 메인 딥페이크 탐지 모델 모듈입니다.
   - **`mymodel/`**: 메인 모델 로직 코드가 위치합니다. (주의: 가상환경 폴더는 Git 관리에서 제외되었습니다.)
   - **`SupCon/`**: Supervised Contrastive Learning (지도 대조 학습) 적용을 위한 학습 코드 및 로그가 포함되어 있습니다.
   - **`idea1_radial_profile/` & `idea1_scatter_plot/`**: 이미지의 주파수 분석(Radial Profile 등) 및 산점도 시각화를 통한 분석 연구 폴더입니다.
   - **`compare/`**: 주파수 대역 분석, GLCM(Gray-Level Co-occurrence Matrix) 개별 분석, 국소 분산(Local Variance) 분석, 경사도 분석(Gradient Analysis) 등 단계별 비교 분석 코드가 위치합니다.
   - **`freq/`**: 주파수 대역별 오차 패턴 시각화 및 분석 실험 결과와 코드들이 포함되어 있습니다.

2. **`DIRE/` (Diffusion Reconstruction Error)**
   - Diffusion Model의 재구성 오차(Reconstruction Error)를 이용해 Generated Image를 탐지하는 프레임워크입니다.
   - guided-diffusion 모델 코드, DIRE 맵 생성 스크립트(`generate_dire.py`), 학습(`train.py`), 평가(`test.py`, `test_dire.py`) 스크립트가 포함되어 있습니다.

3. **`LaRE/` (Language-guided Reconstruction Error)**
   - 텍스트-언어 가이드를 결합한 이미지 재구성 오차 기반의 탐지 프레임워크입니다.
   - LaRE 추출 스크립트(`extract_lare.py`), 리스트 생성 스크립트, 학습/분류기 훈련 스크립트가 포함되어 있습니다.

4. **`FIRE/`**
   - 딥페이크/생성 이미지 탐지를 위한 프레임워크입니다.
   - `train.py`, `eval.py` 및 학습/평가 실행 쉘 스크립트(`train_df.sh`, `eval_df.sh` 등)가 제공됩니다.

5. **`Comparison_imbedding/`**
   - DINOv3 등을 이용하여 임베딩 벡터 간의 비교(`comparison.py`, `Dinov3_Comparsion.py`) 및 시각화(`ex_visual.py`)를 수행하는 분석 도구 모음입니다.

6. **`T_Step/`**
   - DIRE 등 디퓨전 기반 탐지 기법에서 최적의 타임스텝 $T$를 탐색하고 시각화하는 실험 폴더입니다.

---

## 📄 주요 루트 스크립트 (Root Scripts)

- `mixed_dataset.py`: 탐지 모델 학습을 위해 데이터셋을 믹스하는 헬퍼 스크립트.
- `png_to_jpg.py` / `tif_to_png.py`: 데이터 전처리를 위한 이미지 포맷 변환 스크립트.
- `사용법.txt`: DIRE, UFD(CLIP 기반), LaRE 실행을 위한 경로 설정 가이드라인 메모.

---

## 🛠 Git 관리 및 정리 정책 (Git Management Policy)

저장소 크기를 슬림하게 유지하고, 대용량 바이너리 데이터 업로드 오류를 방지하기 위해 다음과 같이 정리되었습니다:

1. **대용량 파일 배제 (`.gitignore`)**:
   - 가상환경 폴더 (`Mymodel/mymodel/`, `Comparison_imbedding/imbedding/` 등) 제외.
   - 모델 가중치 파일 (`*.pth`, `*.ckpt`, `*.pt`, `*.onnx`, `*.dat`, `*.bin`, `*.hash`) 및 캐시/로그 아웃풋 제외.
   - 이미지 데이터셋 및 아카이브 파일 (`*.zip`, `*.tar`, `*.gz` 등) 제외.
2. **서브 디렉터리 `.git` 백업**:
   - 클론해서 가져온 오픈소스 프로젝트들(`DIRE`, `FIRE`, `LaRE` 등)의 내부 `.git` 히스토리를 `.git_backup`으로 안전하게 이름 변경하여, 루트 Git 저장소에서 서브 폴더의 모든 소스 코드와 커스텀 수정 내역을 일반 파일로 완벽히 추적할 수 있도록 구성했습니다.
