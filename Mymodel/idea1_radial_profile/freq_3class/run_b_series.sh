#!/bin/bash
# B 시리즈 주파수 분석 아이디어 순차 실행 스크립트
set -e
cd "$(dirname "$0")"

echo "========================================"
echo "  [B 시리즈] 주파수 분석 아이디어 실행"
echo "========================================"

echo ""
echo "▶ [1/5] B-2: 고주파 대역 Bin Variance 분석"
python b2_hf_variance.py

echo ""
echo "▶ [2/5] B-3: 주파수 대역별 비율(Band Ratio) 분석"
python b3_band_ratio.py

echo ""
echo "▶ [3/5] B-1: 방위각(Azimuthal) 프로파일 분석"
python b1_azimuthal.py

echo ""
echo "▶ [4/5] B-4: 위상(Phase) 일관성 분석"
python b4_phase_consistency.py

echo ""
echo "▶ [5/5] B-compare: 종합 비교"
python b_compare.py

echo ""
echo "========================================"
echo "  ✅ 전체 완료!"
echo "  결과 저장: freq_analysis_outputs/figures/"
echo "========================================"
