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
SAVE_DIR = "./visualization"
os.makedirs(SAVE_DIR, exist_ok=True)

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
    def __init__(self, model_type="dinov3", model_id=None, use_bfloat16=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- DINOv3 model ids 예시 ----
        # (가벼움) facebook/dinov3-vits16-pretrain-lvd1689m
        # (중간)   facebook/dinov3-vitb16-pretrain-lvd1689m
        # (대형)   facebook/dinov3-vitl16-pretrain-lvd1689m
        # (초대형) facebook/dinov3-vit7b16-pretrain-lvd1689m
        # 문서 예시는 vits16을 사용합니다. :contentReference[oaicite:2]{index=2}
        if model_id is None:
            model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"

        self.model_id = model_id
        self.use_bfloat16 = use_bfloat16 and (self.device == "cuda")

        print(f"Loading {model_type.upper()} ({self.model_id}) on {self.device} for Artifact Analysis...")

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

        if self.use_bfloat16:
            # 안전하게 모델만 bf16으로 (입력은 processor가 float32로 만들고, autocast로 커버 가능)
            self.model = self.model.to(dtype=torch.bfloat16)

    def extract_residual_noise(self, image):
        """
        Residual = Original - GaussianBlur(Original)
        """
        gray_image = image.convert("L")
        blurred = gray_image.filter(ImageFilter.GaussianBlur(radius=2))

        np_gray = np.array(gray_image, dtype=np.float32)
        np_blur = np.array(blurred, dtype=np.float32)

        residual = np_gray - np_blur + 127.0
        residual = np.clip(residual, 0, 255).astype(np.uint8)

        return Image.fromarray(residual).convert("RGB")

    def get_artifact_embedding(self, image_path):
        if not os.path.exists(image_path):
            print(f"[Error] File not found: {image_path}")
            return None

        try:
            original_image = Image.open(image_path).convert("RGB")
            residual_image = self.extract_residual_noise(original_image)

            with torch.inference_mode():
                inputs = self.processor(images=residual_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # DINOv3 forward
                outputs = self.model(**inputs)

                # ✅ 권장: pooled_output = "whole-image embedding" :contentReference[oaicite:3]{index=3}
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output  # (1, hidden)
                else:
                    # fallback: CLS token
                    embedding = outputs.last_hidden_state[:, 0, :]  # (1, hidden)

            return F.normalize(embedding.float(), p=2, dim=1).cpu()

        except Exception as e:
            print(f"[Error] Processing {image_path}: {e}")
            return None

# ==========================================
# Chapter 3. 실행 및 벡터 수집 (Execution)
# ==========================================
def main():
    analyzer = ArtifactAnalyzer(
        model_type="dinov3",
        # 필요 시 아래를 vitb16 등으로 바꾸세요.
        model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        use_bfloat16=False,
    )

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

    all_embeddings = torch.cat(embeddings, dim=0)

    # ==========================================
    # Chapter 4. 유사도 계산 및 시각화 (Visualization)
    # ==========================================
    similarity_matrix = all_embeddings @ all_embeddings.T
    sim_matrix_np = similarity_matrix.numpy()

    print("\n--- Artifact Similarity Matrix ---")
    print(sim_matrix_np)

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")

    ax = sns.heatmap(
        sim_matrix_np,
        annot=True,
        fmt=".4f",
        cmap="Purples",
        xticklabels=valid_labels,
        yticklabels=valid_labels,
        vmin=0.0, vmax=1.0,
        cbar_kws={'label': 'Fingerprint Similarity'}
    )

    plt.title("Generative Model Fingerprint Similarity\n(High-Pass Residuals + DINOv3)", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_filename = "artifact_similarity_dinov3.png"
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

