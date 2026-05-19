import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os

def main():
    # ==========================================
    # 1. 설정
    # ==========================================
    test_dir = "/data1/pilot_dataset"
    ckpt_path = "checkpoints/lsun_adm.pth" 
    batch_size = 16
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 사용할 장치: {device}")

    # ==========================================
    # 2. 데이터 로드 (논문 기준 정규화 적용!)
    # ==========================================
    # [수정] DIRE 논문은 0.5, 0.5, 0.5를 사용합니다. 이게 안 맞으면 성능이 확 떨어집니다.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])

    try:
        dataset = datasets.ImageFolder(test_dir, transform=transform)
    except FileNotFoundError:
        print(f"!!! 에러: 폴더를 찾을 수 없습니다: {test_dir}")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f">>> 데이터 로드 완료: 총 {len(dataset)}장")
    print(f">>> 클래스 맵핑: {dataset.class_to_idx}") 
    # {'fake': 0, 'real': 1} 인지 꼭 확인하세요!

    # ==========================================
    # 3. 모델 준비 (다시 1개 출력으로 복귀)
    # ==========================================
    print(">>> 모델 생성 및 가중치 로드 중...")
    model = models.resnet50(pretrained=False)
    
    num_ftrs = model.fc.in_features
    # [복구] 체크포인트 모양([1, 2048])에 맞춰 다시 1로 설정
    model.fc = nn.Linear(num_ftrs, 1) 
    
    model = model.to(device)
    model.eval()

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # 키 처리 로직
        if 'model' in checkpoint: state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint

        # 접두사 제거
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") 
            new_state_dict[name] = v
            
        try:
            # [중요] strict=True로 해서 완벽히 일치하는지 확인
            model.load_state_dict(new_state_dict, strict=True)
            print(">>> 가중치 로드 성공! (완벽 일치)")
        except Exception as e:
            print(f"!!! 경고: {e}")
            # fc 레이어 모양은 이제 맞으니, 다시 strict=False로 시도
            model.load_state_dict(new_state_dict, strict=False)
            print(">>> 일부 불일치가 있지만 로드했습니다.")
    else:
        print(f"!!! 가중치 파일 없음!")

    # ==========================================
    # 4. 테스트 진행 및 디버깅
    # ==========================================
    correct = 0
    total = 0
    
    print("\n>>> 테스트 시작 (상세 로그 출력)...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            
            outputs = model(inputs).squeeze()
            
            # 확률값 계산 (0~1 사이)
            probs = torch.sigmoid(outputs)
            
            # 0.5보다 크면 Real(1), 작으면 Fake(0)
            preds = (probs > 0.5).float()
            
            # [디버깅] 첫 번째 배치의 예측값 눈으로 확인하기
            if i == 0:
                print(f"\n[첫 배치 디버깅]")
                print(f"예측 확률: {probs.cpu().numpy()}")
                print(f"예측 결과: {preds.cpu().numpy()}")
                print(f"실제 정답: {labels.cpu().numpy()}\n")

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"\n========================================")
    print(f"최종 테스트 결과")
    print(f"데이터 개수: {total}장")
    print(f"정확도(Accuracy): {acc:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    main()