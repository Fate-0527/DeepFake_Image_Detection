"""
Phase 0: 공통 피처 추출 파이프라인
- Real / Old Fake / New Fake 레이블 정의
- 각 이미지의 radial profile (orig, err) 추출
- features.pkl 로 캐시 저장
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import glob
import pickle
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm

from fire_model_binary import FIRE_model
from config import Config

# ─── 레이블 정의 ────────────────────────────────────────
LABEL_REAL    = 0
LABEL_OLD_FAKE = 1   # OpenJourney, SD1_4, SD1_5
LABEL_NEW_FAKE = 2   # flux_dev, nano_banana, std_3.5_large_turbo

OLD_FAKE_MODELS = {"OpenJourney", "SD1_4", "SD1_5"}
NEW_FAKE_MODELS = {"flux_dev", "nano_banana", "std_3.5_large_turbo"}

# ─── 설정 ────────────────────────────────────────────────
NUM_SAMPLES_PER_CLASS = 5000       # 클래스당 총 이미지 수
T_STEP          = Config.T_STEP
IMG_SIZE        = Config.IMG_SIZE
CHECKPOINT_PATH = "/data1/checkpoints/HYRE/past_mymodel5_bin_ep66_acc0.8856.pth"
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), '..', 'freq_3class_outputs')
CACHE_PATH      = os.path.join(OUTPUT_DIR, 'features.pkl')
VALID_EXTS      = ('.png', '.jpg', '.jpeg', '.webp')

# Real 소스 폴더들
REAL_DIRS = [
    "/data1/DeepFake/REAL/DIV2K",
    "/data1/DeepFake/REAL/CLIC",
    "/data1/DeepFake/REAL/Flickr2K",
    "/data1/DeepFake/REAL/LSDIR",
    "/data1/DeepFake/REAL/RAISE",
    "/data1/DeepFake/REAL/UCID1338",
]

# Fake 소스 폴더 (모델이름: 경로)
FAKE_DIRS = {
    "OpenJourney":         "/data1/DeepFake/FAKE/OpenJourney",
    "SD1_4":               "/data1/DeepFake/FAKE/SD1_4",
    "SD1_5":               "/data1/DeepFake/FAKE/SD1_5",
    "flux_dev":            "/data1/DeepFake/FAKE/flux_dev",
    "nano_banana":         "/data1/DeepFake/FAKE/nano_banana",
    "std_3.5_large_turbo": "/data1/DeepFake/FAKE/std_3.5_large_turbo",
}


def get_radial_profile(data, center):
    """2D 배열의 중심으로부터 거리(r)에 따른 1D 평균 프로파일."""
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2).astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr   = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def compute_radial_profiles(img_tensor, model, device, t_step):
    """원본 이미지 FFT + FIRE 오차맵 FFT 방사형 프로파일 반환."""
    # 원본 이미지 FFT
    orig_np = img_tensor.squeeze(0).mean(dim=0).cpu().numpy()
    f_shift = np.fft.fftshift(np.fft.fft2(orig_np))
    mag     = np.log(np.abs(f_shift) + 1e-8)
    center  = (mag.shape[0] // 2, mag.shape[1] // 2)
    r_orig  = get_radial_profile(mag, center)

    # FIRE 오차맵 FFT
    with torch.no_grad():
        err_np = model.get_noise_pred_error(img_tensor, t_step=t_step)
        err_np = err_np.squeeze(0).mean(dim=0).cpu().numpy()
    f_err  = np.fft.fftshift(np.fft.fft2(err_np))
    mag_e  = np.log(np.abs(f_err) + 1e-8)
    r_err  = get_radial_profile(mag_e, center)

    return r_orig, r_err


def collect_files(dirs, max_n):
    """주어진 폴더 리스트에서 최대 max_n개 파일 경로 수집."""
    files = []
    for d in dirs:
        fs = sorted([f for f in glob.glob(os.path.join(d, "*.*"))
                     if f.lower().endswith(VALID_EXTS)])
        files.extend(fs)
    return files[:max_n]


def extract_and_save(force=False):
    """피처 추출 메인 함수. force=False면 캐시 존재 시 로드만."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not force and os.path.exists(CACHE_PATH):
        print(f"✅ 캐시 파일 발견: {CACHE_PATH}")
        print("   → 피처 로드 중...")
        with open(CACHE_PATH, 'rb') as f:
            data = pickle.load(f)
        print(f"   → {len(data['labels'])}개 샘플 로드 완료")
        return data

    # ── 모델 로드 ──────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = FIRE_model(device=device)
    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model_sd   = model.state_dict()
        filtered   = {k: v for k, v in state_dict.items()
                      if k in model_sd and v.shape == model_sd[k].shape}
        model.load_state_dict(filtered, strict=False)
        print(f"✅ FIRE 체크포인트 로드 완료")
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    profiles_orig = []
    profiles_err  = []
    labels        = []
    model_names   = []

    def process_files(file_list, label, name_tag):
        for fpath in tqdm(file_list, desc=f"[{name_tag}]", ncols=80):
            try:
                img = transform(Image.open(fpath).convert('RGB')).unsqueeze(0).to(device)
                r_orig, r_err = compute_radial_profiles(img, model, device, T_STEP)
                profiles_orig.append(r_orig)
                profiles_err.append(r_err)
                labels.append(label)
                model_names.append(name_tag)
            except Exception as e:
                pass  # 손상된 파일 스킵

    # ── REAL ──────────────────────────────────────────────
    real_files = collect_files(REAL_DIRS, NUM_SAMPLES_PER_CLASS)
    print(f"\n▶ [REAL] {len(real_files)}장 처리")
    process_files(real_files, LABEL_REAL, "REAL")

    # ── OLD FAKE ──────────────────────────────────────────
    # 모델당 균등 분배
    n_old_per_model = NUM_SAMPLES_PER_CLASS // len(OLD_FAKE_MODELS)
    for model_name in sorted(OLD_FAKE_MODELS):
        folder = FAKE_DIRS[model_name]
        fs = sorted([f for f in glob.glob(os.path.join(folder, "*.*"))
                     if f.lower().endswith(VALID_EXTS)])[:n_old_per_model]
        print(f"\n▶ [OLD FAKE / {model_name}] {len(fs)}장 처리")
        process_files(fs, LABEL_OLD_FAKE, model_name)

    # ── NEW FAKE ──────────────────────────────────────────
    n_new_per_model = NUM_SAMPLES_PER_CLASS // len(NEW_FAKE_MODELS)
    for model_name in sorted(NEW_FAKE_MODELS):
        folder = FAKE_DIRS[model_name]
        fs = sorted([f for f in glob.glob(os.path.join(folder, "*.*"))
                     if f.lower().endswith(VALID_EXTS)])[:n_new_per_model]
        print(f"\n▶ [NEW FAKE / {model_name}] {len(fs)}장 처리")
        process_files(fs, LABEL_NEW_FAKE, model_name)

    # ── 저장 ──────────────────────────────────────────────
    # 길이 통일 (가장 짧은 배열 기준)
    min_len = min(len(p) for p in profiles_orig)
    data = {
        'profiles_orig': np.array([p[:min_len] for p in profiles_orig]),  # [N, L]
        'profiles_err':  np.array([p[:min_len] for p in profiles_err]),   # [N, L]
        'labels':        np.array(labels),                                  # [N]
        'model_names':   model_names,
        'min_len':       min_len,
    }

    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(data, f)

    label_counts = {0: (data['labels']==0).sum(),
                    1: (data['labels']==1).sum(),
                    2: (data['labels']==2).sum()}
    print(f"\n✅ 피처 저장 완료: {CACHE_PATH}")
    print(f"   Real={label_counts[0]}, OldFake={label_counts[1]}, NewFake={label_counts[2]}")
    print(f"   Profile 길이: {min_len}")

    return data


if __name__ == "__main__":
    data = extract_and_save(force=False)
    print("\n[샘플 확인]")
    print(f"  profiles_orig shape: {data['profiles_orig'].shape}")
    print(f"  profiles_err  shape: {data['profiles_err'].shape}")
    print(f"  labels        shape: {data['labels'].shape}")
    print(f"  클래스 분포: Real={( data['labels']==0).sum()}, "
          f"OldFake={(data['labels']==1).sum()}, "
          f"NewFake={(data['labels']==2).sum()}")
