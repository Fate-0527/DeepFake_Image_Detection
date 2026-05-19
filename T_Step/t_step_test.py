import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import glob
import random
from tqdm import tqdm  # 진행상황 표시용

# 사용자 모듈
from fire_model_binary import FIRE_model
from config import Config

# --- 실험 설정 ---
T_STEPS_TO_TEST = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500] # 테스트할 t step 목록
NUM_SAMPLES = 50 # Real 50장, Fake 50장 (총 100장)

def load_image(img_path):
    try:
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
        ])
        img = Image.open(img_path).convert('RGB')
        return transform(img).unsqueeze(0)
    except Exception as e:
        return None

def run_experiment(real_dir, fake_dir, checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 실험 시작! Device: {device}")

    # 1. 모델 로드
    model = FIRE_model(device=device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 모델 로드 완료: {os.path.basename(checkpoint_path)}")
    model.eval()

    # 2. 이미지 리스트 준비 (랜덤 샘플링)
    real_files = sorted(glob.glob(os.path.join(real_dir, "*.*")))
    fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.*")))
    
    # 확장자 필터링
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    real_files = [f for f in real_files if f.lower().endswith(valid_exts)]
    fake_files = [f for f in fake_files if f.lower().endswith(valid_exts)]

    # 샘플링 (파일이 부족하면 있는 만큼만)
    real_files = random.sample(real_files, min(len(real_files), NUM_SAMPLES))
    fake_files = random.sample(fake_files, min(len(fake_files), NUM_SAMPLES))

    print(f"📊 샘플 개수: Real {len(real_files)}장 / Fake {len(fake_files)}장")
    print(f"⏳ 테스트할 T-Step 목록: {T_STEPS_TO_TEST}")

    # 결과를 저장할 리스트
    all_results = []

    # 3. 전수 조사 루프
    # T-Step을 하나씩 바꿔가며 테스트
    for t_val in T_STEPS_TO_TEST:
        print(f"\n[Testing T = {t_val}] 진행 중...")
        
        # (1) Real 이미지 테스트
        for img_path in tqdm(real_files, desc=f"Real (t={t_val})", leave=False):
            img_tensor = load_image(img_path)
            if img_tensor is None: continue
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                # 주파수 필터링
                _, x_pse, _, _ = model.fft_filter_module(img_tensor)
                
                # 에러 계산 (t_step을 인자로 전달)
                err_A = model.get_noise_pred_error(img_tensor, t_step=t_val).mean().item()
                err_B = model.get_noise_pred_error(x_pse, t_step=t_val).mean().item()
                diff = abs(err_A - err_B)

                all_results.append({
                    "type": "Real",
                    "t_step": t_val,
                    "error_A": err_A,
                    "error_B": err_B,
                    "diff": diff,
                    "filename": os.path.basename(img_path)
                })

        # (2) Fake 이미지 테스트
        for img_path in tqdm(fake_files, desc=f"Fake (t={t_val})", leave=False):
            img_tensor = load_image(img_path)
            if img_tensor is None: continue
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                _, x_pse, _, _ = model.fft_filter_module(img_tensor)
                
                err_A = model.get_noise_pred_error(img_tensor, t_step=t_val).mean().item()
                err_B = model.get_noise_pred_error(x_pse, t_step=t_val).mean().item()
                diff = abs(err_A - err_B)

                all_results.append({
                    "type": "Fake",
                    "t_step": t_val,
                    "error_A": err_A,
                    "error_B": err_B,
                    "diff": diff,
                    "filename": os.path.basename(img_path)
                })

    # 4. 결과 정리 및 저장
    df = pd.DataFrame(all_results)
    
    # 저장 폴더
    save_dir = "low_fake_t_step_experiment"
    os.makedirs(save_dir, exist_ok=True)
    
    # (1) 전체 Raw 데이터 저장
    df.to_csv(os.path.join(save_dir, "t_step_raw_data.csv"), index=False)

    # (2) T-Step별 평균 요약표 생성
    summary = df.groupby(['t_step', 'type'])[['error_A', 'error_B', 'diff']].mean().reset_index()
    summary_pivot = summary.pivot(index='t_step', columns='type', values=['error_A', 'error_B', 'diff'])
    
    # 보기 좋게 컬럼 정리
    summary_pivot.columns = [f'{col[1]}_{col[0]}' for col in summary_pivot.columns]
    summary_pivot = summary_pivot.reset_index()
    
    # Diff의 격차(Gap) 계산: Fake_diff가 Real_diff보다 얼마나 큰가?
    summary_pivot['Gap_Ratio'] = summary_pivot['Fake_diff'] / (summary_pivot['Real_diff'] + 1e-9)
    
    csv_path = os.path.join(save_dir, "t_step_summary.csv")
    summary_pivot.to_csv(csv_path, index=False)
    
    print("\n" + "="*50)
    print("📊 T-Step 실험 요약 결과")
    print("="*50)
    print(summary_pivot[['t_step', 'Real_diff', 'Fake_diff', 'Gap_Ratio']].to_string(index=False))
    print("="*50)

    # 5. 그래프 시각화
    plt.figure(figsize=(12, 6))
    
    # Line 1: Real Diff
    plt.plot(summary_pivot['t_step'], summary_pivot['Real_diff'], 
             marker='o', label='Real Image Diff', color='blue', linewidth=2)
    
    # Line 2: Fake Diff
    plt.plot(summary_pivot['t_step'], summary_pivot['Fake_diff'], 
             marker='s', label='Fake Image Diff', color='red', linewidth=2, linestyle='--')
    
    plt.title("Difference Score by T-Step (The higher the Gap, the better)", fontsize=16)
    plt.xlabel("T-Step (Noise Level)", fontsize=14)
    plt.ylabel("Mean Difference (|Error A - Error B|)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    img_path = os.path.join(save_dir, "t_step_analysis.png")
    plt.savefig(img_path)
    print(f"📈 그래프 저장됨: {img_path}")
    print(f"📂 결과 폴더: {save_dir}")

# --- 실행부 ---
MY_CHECKPOINT = "/data1/checkpoints/HYRE/past_mymodel5_bin_ep66_acc0.8856.pth" 

run_experiment(
    real_dir="/data1/DeepFake/REAL/DIV2K",     
    fake_dir="/data1/DeepFake/FAKE/SD1_4", 
    checkpoint_path=MY_CHECKPOINT
)