import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# Chapter 1. 환경 설정 및 경로 (Setup)
# ==========================================
# 결과 저장 경로
SAVE_DIR = "./visualization"
os.makedirs(SAVE_DIR, exist_ok=True)

# 분석할 이미지 경로 (절대 경로)
BASE_PATH = "/data1/DeepFake/FAKE"
TARGET_IMAGES = {
    "Flux Dev": os.path.join(BASE_PATH, "nano_banana", "nano_banana_000036.png"),
    "Nano Banana": os.path.join(BASE_PATH, "nano_banana", "nano_banana_000010.png"),
    "SD 3.5 Large": os.path.join(BASE_PATH, "nano_banana", "nano_banana_000510.png")
}

# ==========================================
# Chapter 2. 아티팩트 추출 및 모델 로드 (Core Logic)
# ==========================================
class ArtifactAnalyzer:
    def __init__(self, model_type="dinov2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_type.upper()} on {self.device} for Artifact Analysis...")
        
        # DINOv2: 텍스처와 구조적 패턴(Artifact) 인식에 최적화된 모델
        self.model_id = "facebook/dinov2-base"
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)

    def extract_residual_noise(self, image):
        """
        이미지에서 '내용'을 제거하고 '노이즈 패턴(아티팩트)'만 남기는 함수
        수식: Residual = Original - GaussianBlur(Original)
        """
        # 1. 그레이스케일 변환 (색상 편향 제거)
        gray_image = image.convert("L")
        
        # 2. 가우시안 블러 (저주파 성분 = 이미지의 내용 추출)
        # radius=2 정도가 내용 제거와 노이즈 보존의 균형이 좋음
        blurred = gray_image.filter(ImageFilter.GaussianBlur(radius=2))
        
        # 3. 차분 계산 (High-Pass Filter)
        np_gray = np.array(gray_image, dtype=np.float32)
        np_blur = np.array(blurred, dtype=np.float32)
        
        # 4. 잔차 추출 및 시각화 보정 (+127)
        # 그냥 빼면 음수가 나오므로 중간값(127)을 더해서 회색조로 만듦
        residual = np_gray - np_blur + 127.0
        
        # 5. 클리핑 및 이미지 변환
        residual = np.clip(residual, 0, 255).astype(np.uint8)
        
        # 6. 모델 입력을 위해 RGB로 재변환 (채널 3개 복사)
        return Image.fromarray(residual).convert("RGB")

    def get_artifact_embedding(self, image_path):
        if not os.path.exists(image_path):
            print(f"[Error] File not found: {image_path}")
            return None
            
        try:
            # 이미지 로드
            original_image = Image.open(image_path).convert("RGB")
            
            # [핵심] 아티팩트(잔차) 이미지 생성
            residual_image = self.extract_residual_noise(original_image)
            
            # (옵션) 추출된 잔차 이미지가 궁금하면 저장해보기
            # debug_name = os.path.basename(image_path)
            # residual_image.save(os.path.join(SAVE_DIR, f"debug_residual_{debug_name}"))

            # DINOv2 추론
            with torch.no_grad():
                inputs = self.processor(images=residual_image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs).last_hidden_state
                
                # CLS Token (이미지 전체의 텍스처 정보 요약)
                embedding = outputs[:, 0, :]
                
            # 정규화 (Cosine Similarity용)
            return F.normalize(embedding, p=2, dim=1).cpu()
            
        except Exception as e:
            print(f"[Error] Processing {image_path}: {e}")
            return None

# ==========================================
# Chapter 3. 실행 및 벡터 수집 (Execution)
# ==========================================
def main():
    analyzer = ArtifactAnalyzer()
    
    embeddings = []
    valid_labels = []

    print("\n--- Starting Artifact Analysis ---")
    for name, path in TARGET_IMAGES.items():
        print(f"Processing: {name}...")
        emb = analyzer.get_artifact_embedding(path)
        
        if emb is not None:
            embeddings.append(emb)
            valid_labels.append(name)

    if len(embeddings) < 2:
        print("비교할 이미지가 충분하지 않습니다.")
        return

    # 벡터 결합 (N, 768)
    all_embeddings = torch.cat(embeddings, dim=0)

    # ==========================================
    # Chapter 4. 유사도 계산 및 시각화 (Visualization)
    # ==========================================
    # 행렬 곱셈으로 코사인 유사도 계산
    similarity_matrix = all_embeddings @ all_embeddings.T
    sim_matrix_np = similarity_matrix.numpy()

    print("\n--- Artifact Similarity Matrix ---")
    print(sim_matrix_np)

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    # 히트맵 생성
    ax = sns.heatmap(
        sim_matrix_np, 
        annot=True, 
        fmt=".4f", 
        cmap="Purples", # 아티팩트 분석이므로 보라색 계열 사용 (차별화)
        xticklabels=valid_labels, 
        yticklabels=valid_labels,
        vmin=0.0, vmax=1.0,
        cbar_kws={'label': 'Fingerprint Similarity'}
    )
    
    plt.title("Generative Model Fingerprint Similarity\n(High-Pass Residuals + DINOv2)", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 결과 저장
    save_filename = "artifact_similarity_final.png"
    save_path = os.path.join(SAVE_DIR, save_filename)
    plt.savefig(save_path, dpi=300)
    
    print(f"\n[Success] Analysis Complete.")
    print(f"Result Image Saved: {save_path}")
    print("-" * 50)
    print("Interpretation Guide:")
    print(" * High Value (close to 1.0): Similar noise patterns (Might share VAE or architecture).")
    print(" * Low Value (close to 0.0): Distinct generative artifacts (Different fingerprints).")

if __name__ == "__main__":
    main()