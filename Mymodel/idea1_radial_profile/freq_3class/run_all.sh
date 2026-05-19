#!/bin/bash
# 주파수 도메인 3진 분류 (A-1 ~ A-5) 전체 실행 스크립트
# 실행: bash run_all.sh [옵션]
#   옵션: --skip-extract   (피처 추출 생략, 캐시 사용)
#          --only-compare  (A-1~A-5 생략, compare_all만 실행)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "  주파수 도메인 3진 분류 전체 파이프라인"
echo "  Real / Old Fake / New Fake"
echo "=================================================="

run_step() {
    echo ""
    echo "──────────────────────────────────────────────────"
    echo "  ▶ $1"
    echo "──────────────────────────────────────────────────"
    python "$2"
}

# Phase 0: 피처 추출
if [[ "$1" != "--skip-extract" && "$1" != "--only-compare" ]]; then
    run_step "Phase 0: 피처 추출 (feature_extractor.py)" feature_extractor.py
fi

if [[ "$1" != "--only-compare" ]]; then
    # Phase 1~5: 각 방법 순서대로
    run_step "Phase 1: A-1 HF Energy Ratio"        a1_hf_ratio.py
    run_step "Phase 2: A-2 Multi-band + SVM/XGBoost" a2_multiband.py
    run_step "Phase 3: A-3 Residual Profile"        a3_residual.py
    run_step "Phase 4: A-4 Spectral Slope"          a4_slope.py
    run_step "Phase 5: A-5 FIRE Error + Orig"       a5_fire_combined.py
fi

# Phase 6: 종합 비교
run_step "Phase 6: 종합 비교 (compare_all.py)"   compare_all.py

echo ""
echo "=================================================="
echo "  ✅ 전체 파이프라인 완료!"
echo "  결과 저장 위치: ../freq_3class_outputs/figures/"
echo "=================================================="
