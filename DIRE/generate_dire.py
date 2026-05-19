import sys
import os
# 현재 경로를 강제로 추가
sys.path.append(os.getcwd())

import torch
import traceback
import torchvision.transforms as T
from PIL import Image
from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion

def main():
    # ==========================================
    # 1. 경로 설정 (사용자 지정 경로 유지)
    # ==========================================
    datasets = [
        {
            "name": "Fake 데이터 (ADM)",
            "input_path": ["/data1/Deepfake/FAKE"],
            "output_path": "/data1/Deepfake/RE_FAKE"
        },
        {
            "name": "Real 데이터 (LSUN)",
            "input_path": ["/data1/Deepfake/REAL"],
            "output_path": "/data1/Deepfake/RE_REAL"
        }
    ]
    
    ckpt_path = "checkpoints/lsun_bedroom.pt"

    # ==========================================
    # 2. 재구성 모델(ADM) 로드 - [핵심 수정!]
    # ==========================================
    print(">>> [1/3] 재구성 모델(ADM) 로드 중 (논문 공식: DDIM 20 Steps)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = model_and_diffusion_defaults()
    config.update({
        "image_size": 256,
        "num_channels": 256,
        "num_res_blocks": 2,
        "num_heads": 4,
        "attention_resolutions": "32,16,8",
        "dropout": 0.0,
        "learn_sigma": True,
        "class_cond": False,
        "diffusion_steps": 1000,
        "noise_schedule": "linear",
        # ▼▼▼ [중요] 논문 구현의 핵심! 1000번을 20번으로 압축하는 설정 ▼▼▼
        "timestep_respacing": "ddim20", 
        # ▲▲▲ 이 줄이 없으면 1000번 다 돌아서 성능이 떨어집니다. ▲▲▲
        "use_scale_shift_norm": True,
        "resblock_updown": True
    })
    
    model, diffusion = create_model_and_diffusion(**config)
    
    if not os.path.exists(ckpt_path):
        print(f"!!! [치명적 에러] 가중치 파일이 없습니다: {ckpt_path}")
        return

    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        print(">>> 모델 로드 성공!")
    except Exception:
        print("!!! 모델 로드 중 에러 발생 !!!")
        traceback.print_exc()
        return

    # 이미지 전처리 (논문 표준: 0.5)
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # ==========================================
    # 3. 변환 루프 실행
    # ==========================================
    for data_info in datasets:
        name = data_info["name"]
        in_dirs = data_info["input_path"]
        out_dir = data_info["output_path"]

        print(f"\n========================================")
        print(f"작업 시작: {name}")
        print(f"입력: {in_dirs}")
        print(f"출력: {out_dir}")
        print(f"========================================")

        # input_path가 str이면 리스트로 변환
        if isinstance(in_dirs, str):
            in_dirs = [in_dirs]

        all_images = []
        for in_dir in in_dirs:
            if not os.path.exists(in_dir):
                print(f"!!! 에러: 입력 폴더가 존재하지 않습니다 ({in_dir})")
                continue
            for root, _, files in os.walk(in_dir):
                images = [os.path.join(root, f) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.extend(images)

        os.makedirs(out_dir, exist_ok=True)
        total_imgs = len(all_images)
        print(f">>> 총 {total_imgs}장의 이미지를 변환합니다.")

        for idx, img_path in enumerate(all_images):
            img_name = os.path.basename(img_path)
            try:
                # A. 이미지 로드
                img = Image.open(img_path).convert("RGB")
                x0 = transform(img).unsqueeze(0).to(device)

                # B. DDIM Inversion & Reconstruction
                with torch.no_grad():
                    # 1. Inversion
                    x_inv = diffusion.ddim_reverse_sample_loop(
                        model, x0.shape, noise=x0, clip_denoised=True, device=device
                    )

                    # 2. Reconstruction
                    x_recon = diffusion.ddim_sample_loop(
                        model, x0.shape, noise=x_inv, clip_denoised=True, device=device
                    )

                # C. DIRE 계산 및 저장 - [수식 수정!]
                abs_diff = torch.abs(x0 - x_recon)
                dire_tensor = abs_diff.clamp(0, 1) 
                
                save_name = os.path.join(out_dir, img_name)
                T.ToPILImage()(dire_tensor.squeeze(0).cpu()).save(save_name)

                if (idx + 1) % 10 == 0:
                    print(f"[{name}] 진행 중... {idx + 1}/{total_imgs}")

            except Exception:
                print(f"\n!!! 에러 발생 (파일: {img_name}) !!!")
                traceback.print_exc()
                continue

    print("\n>>> 모든 변환 작업이 완료되었습니다!")
    print(">>> 이제 test_dire_final.py의 경로를 'adm_re'와 'real_re'로 수정하고 돌려보세요!")

if __name__ == "__main__":
    main()