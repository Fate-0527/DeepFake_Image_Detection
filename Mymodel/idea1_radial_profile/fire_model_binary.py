from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler # ★ 추가됨
from typing import Literal

# ---------------------------------------------------------
# 1. Backbone: Frequency-aware ResNet50 (기존 동일)
# ---------------------------------------------------------
def get_frq_resnet_model(mode: Literal["rgb", "ours", "frq"], norm_layer: Literal["batch", "instance"], pretrained: bool) -> nn.Module:
    if norm_layer == "batch":
        norm = nn.BatchNorm2d
    elif norm_layer == "instance":
        norm = nn.InstanceNorm2d
    else:
        raise AssertionError("Unknown norm layer")

    if norm == nn.InstanceNorm2d:
        model = torchvision.models.resnet50(num_classes=1000, pretrained=False, norm_layer=norm)
    else:
        model = torchvision.models.resnet50(weights="IMAGENET1K_V2", norm_layer=norm)

    # 입력 채널 수정 (6채널: Error_A(3) + Error_B(3))
    # ★ 설명: ResNet이 6채널을 받으면 첫 번째 Conv 레이어에서 스스로 (w1*A + w2*B) 연산을 수행합니다.
    # 즉, 굳이 밖에서 A-B를 계산해서 3채널로 줄여 넣는 것보다,
    # 6채널을 다 넣어주면 모델이 알아서 "차이(Difference)"를 학습합니다.
    if mode == "frq":
        original_weights = model.conv1.weight.data
        new_weights = torch.cat([original_weights * 0.5] * 2, dim=1) 
        model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight.data = new_weights

    # ★ 2-Class Softmax 방식 (threshold 불필요)
    # 출력: [B, 2] → class 0: Real, class 1: Fake
    # Loss: CrossEntropyLoss 사용
    # Inference: argmax로 판정 (더 큰 logit이 예측 클래스)
    model.fc = nn.Linear(2048, 2)
    nn.init.normal_(model.fc.weight.data, 0.0, 0.02)

    return model

# ---------------------------------------------------------
# 2. ESPCN (Mask Autoencoder) (기존 동일)
# ---------------------------------------------------------
class ESPCN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, channels=64, upscale_factor=1):
        super(ESPCN, self).__init__()
        hidden_channels = channels // 2
        out_channels = int(out_channels * (upscale_factor ** 2))
        self.bn = nn.BatchNorm2d(in_channels)
        
        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, channels, (5, 5), (1, 1), (2, 2)),
            nn.Tanh(),
            nn.Conv2d(channels, hidden_channels, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),
        )

        self.sub_pixel_0 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            nn.Sigmoid(),
        )

        self.sub_pixel_1 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.feature_maps(x)
        mask_mid = self.sub_pixel_0(x)
        mask_mid_c = self.sub_pixel_1(x)
        return mask_mid, mask_mid_c

# ---------------------------------------------------------
# 3. FMRE Implementation (fft_filter) (기존 동일)
# ---------------------------------------------------------
class fft_filter(nn.Module):
    def __init__(self, radiuslow=40, radiushigh=120, rows=None, cols=None):
        super(fft_filter, self).__init__()
        self.radiuslow = radiuslow
        self.radiushigh = radiushigh
        # ★ Config.IMG_SIZE 사용 (None이면 기본값)
        self.rows = rows if rows is not None else Config.IMG_SIZE
        self.cols = cols if cols is not None else Config.IMG_SIZE
        
        i_mask, r_i_mask = self.init_mask()
        self.register_buffer('i_mask', i_mask)
        self.register_buffer('r_i_mask', r_i_mask)
        
        self.mask_autoencoder = ESPCN(in_channels=3, out_channels=1, channels=64, upscale_factor=1)
    
    def init_mask(self):
        # ★ 수정: (1, 1, H, W) 형태로 생성 (배치 브로드캐스팅 지원)
        mask = torch.zeros((1, 1, self.rows, self.cols), dtype=torch.float32)
        crow, ccol = self.rows // 2 , self.cols // 2
        y, x = torch.meshgrid(torch.arange(self.rows), torch.arange(self.cols), indexing='ij')
        dist_sq = (x - ccol) ** 2 + (y - crow) ** 2
        mask_area = (dist_sq >= self.radiuslow**2) & (dist_sq < self.radiushigh**2)
        mask[:, :, mask_area] = 1.0
        return mask, 1-mask

    def middle_pass_filter(self, image):
        freq_image = torch.fft.fftn(image * 255, dim=(-2, -1))
        freq_image = torch.fft.fftshift(freq_image, dim=(-2, -1))
        
        magnitude = torch.log(torch.abs(freq_image) + 1e-7) / 20.0 
        mask_mid_frq, mask_mid_filterd = self.mask_autoencoder(magnitude)
        
        # 1. 중간 주파수만 남긴 이미지 (x_mid)
        middle_freq = freq_image * mask_mid_frq
        x_mid = torch.abs(torch.fft.ifftn(torch.fft.ifftshift(middle_freq, dim=(-2, -1)), dim=(-2, -1))) / 255.0
        
        # 2. 중간 주파수를 제거한 이미지 (x_pse)
        middle_filtered = freq_image * mask_mid_filterd
        x_pse_raw = torch.abs(torch.fft.ifftn(torch.fft.ifftshift(middle_filtered, dim=(-2, -1)), dim=(-2, -1)))
        
        # ★ [수정] L2 Normalization 적용 (기존 / 255.0 대신)
        # 이미지(Batch, C, H, W)를 벡터로 보고 L2 Norm으로 나눔
        B, C, H, W = x_pse_raw.shape
        flat_x = x_pse_raw.view(B, -1) # 평탄화
        norm = torch.norm(flat_x, p=2, dim=1).view(B, 1, 1, 1) # L2 Norm 계산
        
        # 0으로 나누기 방지 (epsilon)
        x_pse = x_pse_raw / (norm + 1e-8)

        return x_mid, x_pse, mask_mid_frq, mask_mid_filterd

    def forward(self, image):
        return self.middle_pass_filter(image)

# ---------------------------------------------------------
# 4. Final DiffLaRE Model (구 FIRE_model)
# ---------------------------------------------------------
class FIRE_model(nn.Module):
    def __init__(self, mode="frq", norm_layer="instance", pretrained=True, radiuslow=40, radiushigh=120, device="cuda"):
        super(FIRE_model, self).__init__()
        self.device = device
        
        # ★ 수정: 캐시 경로를 /data1 내부로 지정
        # (폴더가 없으면 알아서 생성하고 다운로드 받습니다)
        hf_cache_dir = "/data1/huggingface_cache"
        
        print(f">>> Loading Stable Diffusion 1.5 Components from {hf_cache_dir}...")
        
        # 1. VAE 로드 (cache_dir 추가)
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="vae", 
            cache_dir=hf_cache_dir
        ).to(device)
        
        # 2. U-Net 로드 (cache_dir 추가)
        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="unet", 
            cache_dir=hf_cache_dir
        ).to(device)
        
        # 3. Scheduler 로드 (cache_dir 추가)
        self.scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="scheduler", 
            cache_dir=hf_cache_dir
        )
        
        # 모델 얼리기 (학습 X)
        self.vae.eval()
        self.unet.eval()
        for param in self.vae.parameters(): param.requires_grad = False
        for param in self.unet.parameters(): param.requires_grad = False
            
        # 4. Classifier
        # ★ [핵심 2] Latent 입력용 ResNet50 (Upscale X)
        self.resnet = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        
        # 3채널 -> 8채널 (Error_A 4ch + Error_B 4ch)
        original_weights = self.resnet.conv1.weight.data
        new_weights = torch.cat([
            original_weights,               # 1~3ch
            original_weights,               # 4~6ch
            original_weights[:, :2, :, :]   # 7~8ch
        ], dim=1)
        
        self.resnet.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.conv1.weight.data = new_weights
        self.resnet.fc = nn.Linear(2048, 2)
        
        # 5. FMRE Module
        self.fft_filter_module = fft_filter(radiuslow=radiuslow, radiushigh=radiushigh).to(device)

    def get_noise_pred_error(self, image, t_step=100):
        """★ LaRE 핵심: 노이즈 예측 오차 계산 함수"""
        with torch.no_grad():
            # 1. Latent 변환 (Image: [B, 3, 256, 256] -> Latent: [B, 4, 32, 32])
            latents = self.vae.encode(image * 2.0 - 1.0).latent_dist.mode() * 0.18215
            
            # 2. 노이즈 추가
            noise = torch.randn_like(latents).to(self.device)
            timesteps = torch.tensor([t_step], device=self.device).long()
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
            
            # 3. U-Net 예측
            encoder_hidden_states = torch.zeros((image.shape[0], 77, 768)).to(self.device)
            noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            
            # 4. Latent Space 오차 계산 [B, 4, 32, 32]
            error_latent = torch.abs(noise - noise_pred)

            return error_latent

    def forward(self, x):
        # 1. Frequency Filtering
        # x_mid: 중간 주파수 이미지 (Loss 계산용)
        # x_pse: 중간 주파수가 제거된 이미지 (B 계산용)
        x_mid, x_pse, m_mid, m_mid_c = self.fft_filter_module(x)

        # 2. LaRE Error Calculation
        # Error A: 원본 이미지의 노이즈 예측 오차
        error_A = self.get_noise_pred_error(x, t_step=Config.T_STEP)
        
        # Error B: 주파수 제거 이미지의 노이즈 예측 오차
        error_B = self.get_noise_pred_error(x_pse, t_step=Config.T_STEP)

        # 3. Classifier Forward
        # [Error A, Error B]를 합쳐서(Concat) 6채널로 넣습니다.
        # ResNet의 첫 Conv 레이어가 학습을 통해 (Error A - Error B)의 특징을 찾아냅니다.
        features = torch.cat([error_A, error_B], dim=1)

        out = self.resnet(features)

        x_mid_norm = x_mid * 2.0 - 1.0 # [0,1] -> [-1,1]
        x_mid_latent_dist = self.vae.encode(x_mid_norm).latent_dist
        x_mid_latent = x_mid_latent_dist.mode() * 0.18215 # [Batch, 4, 64, 64]

        # 참고: Loss 계산을 위해 error_A(delta_x)를 반환
        return out, x_mid_latent, error_A, m_mid, m_mid_c
    