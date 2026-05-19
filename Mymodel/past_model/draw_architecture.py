"""
DiffLaRE Model Architecture Diagram Generator
논문용 모델 구조도 이미지 생성 스크립트
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def draw_box(ax, x, y, width, height, text, color='lightblue', text_color='black', fontsize=9, alpha=0.8):
    """박스와 텍스트를 그리는 함수"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='black', linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            color=text_color, fontweight='bold', wrap=True)
    return box

def draw_arrow(ax, start, end, color='black', style='->', connectionstyle="arc3,rad=0"):
    """화살표를 그리는 함수"""
    arrow = FancyArrowPatch(start, end, arrowstyle=style, color=color,
                            mutation_scale=15, linewidth=1.5,
                            connectionstyle=connectionstyle)
    ax.add_patch(arrow)
    return arrow

def draw_difflaare_architecture():
    """DiffLaRE 전체 아키텍처를 그리는 함수"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 색상 정의
    colors = {
        'input': '#FFE4B5',      # 입력 (연한 주황)
        'fmre': '#98FB98',       # FMRE (연한 초록)
        'lare': '#87CEEB',       # LaRE (연한 파랑)
        'vae': '#DDA0DD',        # VAE (연한 보라)
        'unet': '#F0E68C',       # U-Net (연한 노랑)
        'error': '#FFB6C1',      # Error (연한 핑크)
        'classifier': '#FFA07A', # Classifier (연한 살몬)
        'output': '#90EE90',     # Output (연한 초록)
        'loss': '#D3D3D3',       # Loss (연한 회색)
    }
    
    # ========== 타이틀 ==========
    ax.text(8, 19.5, 'DiffLaRE: Diffusion-based Latent Reconstruction Error\nfor Deepfake Detection', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # ========== Stage 0: Input ==========
    draw_box(ax, 8, 18, 2.5, 0.7, 'Input Image x\n[B, 3, H, W]', colors['input'], fontsize=8)
    
    # ========== Stage 1: FMRE Module ==========
    # FMRE 배경 박스
    fmre_bg = FancyBboxPatch((0.5, 14.8), 5.5, 2.8, boxstyle="round,pad=0.1",
                              facecolor='#E8F5E9', edgecolor='green', linewidth=2, alpha=0.3)
    ax.add_patch(fmre_bg)
    ax.text(3.25, 17.4, 'Stage 1: FMRE Module', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='darkgreen')
    
    draw_box(ax, 3.25, 16.5, 2, 0.6, 'FFT Transform', colors['fmre'], fontsize=8)
    draw_box(ax, 3.25, 15.7, 2, 0.6, 'ESPCN\n(Mask AutoEncoder)', colors['fmre'], fontsize=7)
    draw_box(ax, 2, 15, 1.5, 0.5, 'x_mid', colors['fmre'], fontsize=8)
    draw_box(ax, 4.5, 15, 1.5, 0.5, 'x_pse', colors['fmre'], fontsize=8)
    
    # FMRE 내부 화살표
    draw_arrow(ax, (3.25, 16.2), (3.25, 16.0))
    draw_arrow(ax, (2.5, 15.4), (2, 15.25))
    draw_arrow(ax, (4, 15.4), (4.5, 15.25))
    
    # ========== Stage 2: LaRE Extraction ==========
    # LaRE A 배경 박스 (원본)
    lare_a_bg = FancyBboxPatch((0.5, 10.2), 5.5, 4.3, boxstyle="round,pad=0.1",
                                facecolor='#E3F2FD', edgecolor='blue', linewidth=2, alpha=0.3)
    ax.add_patch(lare_a_bg)
    ax.text(3.25, 14.3, 'Stage 2-A: LaRE (Original x)', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='darkblue')
    
    # LaRE B 배경 박스 (x_pse)
    lare_b_bg = FancyBboxPatch((10, 10.2), 5.5, 4.3, boxstyle="round,pad=0.1",
                                facecolor='#FFF3E0', edgecolor='orange', linewidth=2, alpha=0.3)
    ax.add_patch(lare_b_bg)
    ax.text(12.75, 14.3, 'Stage 2-B: LaRE (x_pse)', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='darkorange')
    
    # LaRE A 컴포넌트
    draw_box(ax, 3.25, 13.5, 2.2, 0.6, 'VAE Encoder\n(SD 1.5)', colors['vae'], fontsize=7)
    draw_box(ax, 3.25, 12.6, 2.2, 0.6, 'Add Noise ε\n(DDPM Forward)', colors['lare'], fontsize=7)
    draw_box(ax, 3.25, 11.7, 2.2, 0.6, 'U-Net\n(Predict ε̂)', colors['unet'], fontsize=7)
    draw_box(ax, 3.25, 10.8, 2.2, 0.6, '|ε - ε̂| → Upscale', colors['error'], fontsize=7)
    
    # LaRE B 컴포넌트
    draw_box(ax, 12.75, 13.5, 2.2, 0.6, 'VAE Encoder\n(SD 1.5)', colors['vae'], fontsize=7)
    draw_box(ax, 12.75, 12.6, 2.2, 0.6, 'Add Noise ε\n(DDPM Forward)', colors['lare'], fontsize=7)
    draw_box(ax, 12.75, 11.7, 2.2, 0.6, 'U-Net\n(Predict ε̂)', colors['unet'], fontsize=7)
    draw_box(ax, 12.75, 10.8, 2.2, 0.6, '|ε - ε̂| → Upscale', colors['error'], fontsize=7)
    
    # LaRE 내부 화살표
    for x in [3.25, 12.75]:
        draw_arrow(ax, (x, 13.2), (x, 12.9))
        draw_arrow(ax, (x, 12.3), (x, 12.0))
        draw_arrow(ax, (x, 11.4), (x, 11.1))
    
    # Error 출력 박스
    draw_box(ax, 3.25, 9.8, 2, 0.6, 'Error_A (Δx)\n[B, 3, H, W]', colors['error'], fontsize=7)
    draw_box(ax, 12.75, 9.8, 2, 0.6, 'Error_B (Δx_pse)\n[B, 3, H, W]', colors['error'], fontsize=7)
    
    # ========== Stage 3: Classification ==========
    # Classifier 배경 박스
    cls_bg = FancyBboxPatch((4.5, 5.5), 7, 3.8, boxstyle="round,pad=0.1",
                             facecolor='#FFEBEE', edgecolor='red', linewidth=2, alpha=0.3)
    ax.add_patch(cls_bg)
    ax.text(8, 9.1, 'Stage 3: Classification', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='darkred')
    
    draw_box(ax, 8, 8.3, 3, 0.6, 'Concatenate\n[Error_A || Error_B] → [B, 6, H, W]', colors['classifier'], fontsize=7)
    draw_box(ax, 8, 7.4, 2.5, 0.5, 'BatchNorm2d(6)', colors['classifier'], fontsize=8)
    draw_box(ax, 8, 6.5, 3, 0.7, 'ResNet50\n(6-channel input)', colors['classifier'], fontsize=8)
    draw_box(ax, 8, 5.7, 2.5, 0.5, 'FC: 2048 → 2', colors['classifier'], fontsize=8)
    
    # Classifier 내부 화살표
    draw_arrow(ax, (8, 8.0), (8, 7.65))
    draw_arrow(ax, (8, 7.15), (8, 6.85))
    draw_arrow(ax, (8, 6.15), (8, 5.95))
    
    # ========== Output ==========
    draw_box(ax, 8, 4.8, 2.5, 0.6, 'Output: [B, 2]\nReal vs Fake', colors['output'], fontsize=8)
    
    # ========== Loss Functions ==========
    draw_box(ax, 3, 4, 2, 0.5, 'L_ce\n(CrossEntropy)', colors['loss'], fontsize=7)
    draw_box(ax, 8, 4, 2, 0.5, 'L_mid_rec\n(MSE)', colors['loss'], fontsize=7)
    draw_box(ax, 13, 4, 2, 0.5, 'L_mask\n(Diversity)', colors['loss'], fontsize=7)
    
    # ========== 전체 연결 화살표 ==========
    # Input → FMRE, LaRE A
    draw_arrow(ax, (8, 17.65), (8, 17.2), style='-|>')
    ax.annotate('', xy=(3.25, 17.2), xytext=(8, 17.2),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.5))
    draw_arrow(ax, (3.25, 17.2), (3.25, 16.8))
    
    # Input → LaRE A (직접 연결)
    ax.annotate('', xy=(8, 13.8), xytext=(8, 17.2),
                arrowprops=dict(arrowstyle='-|>', color='gray', lw=1, ls='--'))
    ax.annotate('', xy=(3.25, 13.8), xytext=(8, 13.8),
                arrowprops=dict(arrowstyle='-|>', color='gray', lw=1, ls='--'))
    
    # x_pse → LaRE B
    ax.annotate('', xy=(12.75, 13.8), xytext=(4.5, 14.75),
                arrowprops=dict(arrowstyle='-|>', color='orange', lw=1.5,
                               connectionstyle="arc3,rad=-0.2"))
    
    # LaRE outputs → Concat
    draw_arrow(ax, (3.25, 10.5), (3.25, 10.05))
    draw_arrow(ax, (12.75, 10.5), (12.75, 10.05))
    ax.annotate('', xy=(8, 8.6), xytext=(3.25, 9.5),
                arrowprops=dict(arrowstyle='-|>', color='blue', lw=1.5,
                               connectionstyle="arc3,rad=0.2"))
    ax.annotate('', xy=(8, 8.6), xytext=(12.75, 9.5),
                arrowprops=dict(arrowstyle='-|>', color='orange', lw=1.5,
                               connectionstyle="arc3,rad=-0.2"))
    
    # Classifier → Output
    draw_arrow(ax, (8, 5.45), (8, 5.1))
    
    # Output → Loss
    ax.annotate('', xy=(3, 4.25), xytext=(8, 4.5),
                arrowprops=dict(arrowstyle='-|>', color='gray', lw=1, ls='--'))
    draw_arrow(ax, (8, 4.5), (8, 4.25), color='gray', style='-|>')
    ax.annotate('', xy=(13, 4.25), xytext=(8, 4.5),
                arrowprops=dict(arrowstyle='-|>', color='gray', lw=1, ls='--'))
    
    # x_mid → L_mid_rec 연결
    ax.annotate('', xy=(8, 4.25), xytext=(2, 14.75),
                arrowprops=dict(arrowstyle='-|>', color='green', lw=1, ls=':',
                               connectionstyle="arc3,rad=0.3"))
    
    # ========== 범례 ==========
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], edgecolor='black', label='Input'),
        mpatches.Patch(facecolor=colors['fmre'], edgecolor='black', label='FMRE (Frequency)'),
        mpatches.Patch(facecolor=colors['vae'], edgecolor='black', label='VAE Encoder'),
        mpatches.Patch(facecolor=colors['lare'], edgecolor='black', label='DDPM Forward'),
        mpatches.Patch(facecolor=colors['unet'], edgecolor='black', label='U-Net'),
        mpatches.Patch(facecolor=colors['error'], edgecolor='black', label='Error Map'),
        mpatches.Patch(facecolor=colors['classifier'], edgecolor='black', label='Classifier'),
        mpatches.Patch(facecolor=colors['output'], edgecolor='black', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, ncol=2)
    
    # ========== Shape 정보 텍스트 ==========
    ax.text(15.5, 18, 'Shape Flow:\n[B,3,H,W]\n    ↓ VAE\n[B,4,h,w]\n    ↓ U-Net\n[B,4,h,w]\n    ↓ Upscale\n[B,3,H,W]', 
            ha='left', va='top', fontsize=7, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def draw_simplified_architecture():
    """간소화된 가로형 구조도"""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    colors = {
        'input': '#FFE4B5',
        'fmre': '#98FB98',
        'lare': '#87CEEB',
        'classifier': '#FFA07A',
        'output': '#90EE90',
    }
    
    # 타이틀
    ax.text(9, 7.5, 'DiffLaRE Architecture (Simplified)', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Input
    draw_box(ax, 1.5, 4, 2, 1.2, 'Input\nx', colors['input'], fontsize=10)
    
    # FMRE
    fmre_bg = FancyBboxPatch((2.8, 2.5), 2.5, 3, boxstyle="round,pad=0.1",
                              facecolor='#E8F5E9', edgecolor='green', linewidth=2, alpha=0.3)
    ax.add_patch(fmre_bg)
    ax.text(4.05, 5.3, 'FMRE', ha='center', fontsize=9, fontweight='bold', color='darkgreen')
    draw_box(ax, 4.05, 4.5, 1.8, 0.6, 'FFT', colors['fmre'], fontsize=8)
    draw_box(ax, 4.05, 3.7, 1.8, 0.6, 'ESPCN', colors['fmre'], fontsize=8)
    draw_box(ax, 4.05, 2.9, 1.8, 0.5, 'x_pse', colors['fmre'], fontsize=8)
    
    # LaRE Path A (상단)
    lare_a_bg = FancyBboxPatch((5.8, 4.8), 4.5, 2, boxstyle="round,pad=0.1",
                                facecolor='#E3F2FD', edgecolor='blue', linewidth=2, alpha=0.3)
    ax.add_patch(lare_a_bg)
    ax.text(8, 6.6, 'LaRE (x)', ha='center', fontsize=9, fontweight='bold', color='darkblue')
    draw_box(ax, 6.8, 5.5, 1.5, 0.8, 'VAE\nEnc', '#DDA0DD', fontsize=7)
    draw_box(ax, 8.3, 5.5, 1.5, 0.8, 'U-Net\nε̂', '#F0E68C', fontsize=7)
    draw_box(ax, 9.8, 5.5, 1.2, 0.8, '|ε-ε̂|', '#FFB6C1', fontsize=7)
    
    # LaRE Path B (하단)
    lare_b_bg = FancyBboxPatch((5.8, 1.2), 4.5, 2, boxstyle="round,pad=0.1",
                                facecolor='#FFF3E0', edgecolor='orange', linewidth=2, alpha=0.3)
    ax.add_patch(lare_b_bg)
    ax.text(8, 3, 'LaRE (x_pse)', ha='center', fontsize=9, fontweight='bold', color='darkorange')
    draw_box(ax, 6.8, 2, 1.5, 0.8, 'VAE\nEnc', '#DDA0DD', fontsize=7)
    draw_box(ax, 8.3, 2, 1.5, 0.8, 'U-Net\nε̂', '#F0E68C', fontsize=7)
    draw_box(ax, 9.8, 2, 1.2, 0.8, '|ε-ε̂|', '#FFB6C1', fontsize=7)
    
    # Concat
    draw_box(ax, 11.5, 4, 1.5, 1.5, 'Concat\n[6ch]', '#D3D3D3', fontsize=8)
    
    # Classifier
    cls_bg = FancyBboxPatch((12.5, 2.8), 3, 2.4, boxstyle="round,pad=0.1",
                             facecolor='#FFEBEE', edgecolor='red', linewidth=2, alpha=0.3)
    ax.add_patch(cls_bg)
    ax.text(14, 5, 'Classifier', ha='center', fontsize=9, fontweight='bold', color='darkred')
    draw_box(ax, 14, 4, 2.2, 0.7, 'ResNet50', colors['classifier'], fontsize=9)
    draw_box(ax, 14, 3.2, 2.2, 0.5, 'FC → 2', colors['classifier'], fontsize=9)
    
    # Output
    draw_box(ax, 16.5, 4, 1.5, 1.2, 'Output\nReal/Fake', colors['output'], fontsize=9)
    
    # 화살표들
    draw_arrow(ax, (2.5, 4), (2.8, 4))  # Input → FMRE
    draw_arrow(ax, (2.5, 4.5), (5.8, 5.5), connectionstyle="arc3,rad=0.2")  # Input → LaRE A
    draw_arrow(ax, (5.3, 2.9), (5.8, 2))  # FMRE → LaRE B
    
    # LaRE A 내부
    draw_arrow(ax, (7.55, 5.5), (7.55, 5.5))
    draw_arrow(ax, (9.05, 5.5), (9.2, 5.5))
    
    # LaRE B 내부  
    draw_arrow(ax, (7.55, 2), (7.55, 2))
    draw_arrow(ax, (9.05, 2), (9.2, 2))
    
    # LaRE → Concat
    draw_arrow(ax, (10.4, 5.5), (10.75, 4.5), connectionstyle="arc3,rad=-0.2")
    draw_arrow(ax, (10.4, 2), (10.75, 3.5), connectionstyle="arc3,rad=0.2")
    
    # Concat → Classifier → Output
    draw_arrow(ax, (12.25, 4), (12.5, 4))
    draw_arrow(ax, (15.5, 4), (15.75, 4))
    
    # 수식 추가
    ax.text(8, 0.6, r'$\mathbf{Error} = |\epsilon - \hat{\epsilon}|$  where  $\hat{\epsilon} = U_\theta(z_t, t)$', 
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    return fig


def draw_lare_detail():
    """LaRE 모듈 상세 구조도"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 타이틀
    ax.text(6, 9.5, 'LaRE Module: Latent Reconstruction Error', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Input
    draw_box(ax, 2, 8, 2.5, 0.8, 'Image x\n[B, 3, H, W]', '#FFE4B5', fontsize=9)
    
    # VAE Encoder
    draw_box(ax, 2, 6.5, 2.5, 1, 'VAE Encoder\n(SD 1.5)\nx*2-1 → latent', '#DDA0DD', fontsize=8)
    
    # Latent z0
    draw_box(ax, 2, 5, 2, 0.7, 'z₀\n[B, 4, 32, 32]', '#E3F2FD', fontsize=9)
    
    # Noise
    draw_box(ax, 6, 6.5, 2, 0.8, 'Noise ε\n~ N(0, I)', '#FFB6C1', fontsize=9)
    
    # DDPM Forward
    draw_box(ax, 4, 4, 4, 1, 'DDPM Forward Process\nzₜ = √ᾱₜ·z₀ + √(1-ᾱₜ)·ε\n(t = T_STEP)', '#87CEEB', fontsize=8)
    
    # Noisy Latent
    draw_box(ax, 4, 2.8, 2.5, 0.7, 'zₜ (Noisy Latent)\n[B, 4, 32, 32]', '#E3F2FD', fontsize=8)
    
    # U-Net
    draw_box(ax, 8, 4, 3, 1.2, 'U-Net (SD 1.5)\nε̂ = Uθ(zₜ, t)\n(Frozen)', '#F0E68C', fontsize=8)
    
    # Predicted Noise
    draw_box(ax, 8, 2.5, 2, 0.7, 'ε̂ (Predicted)\n[B, 4, 32, 32]', '#FFB6C1', fontsize=8)
    
    # Error Computation
    draw_box(ax, 6, 1.3, 3.5, 0.8, 'Error = |ε - ε̂|\n[B, 4, 32, 32]', '#FF6B6B', fontsize=9, text_color='white')
    
    # Upscale & Output
    draw_box(ax, 6, 0.3, 3.5, 0.7, 'Upscale → [B, 3, H, W]\n(Bilinear + Channel Slice)', '#90EE90', fontsize=8)
    
    # 화살표
    draw_arrow(ax, (2, 7.6), (2, 7.0))
    draw_arrow(ax, (2, 6.0), (2, 5.35))
    draw_arrow(ax, (3, 5), (4, 4.5), connectionstyle="arc3,rad=-0.1")
    draw_arrow(ax, (6, 6.1), (5, 4.5), connectionstyle="arc3,rad=0.1")
    draw_arrow(ax, (4, 3.5), (4, 3.15))
    draw_arrow(ax, (5.25, 2.8), (6.5, 4), connectionstyle="arc3,rad=-0.3")
    draw_arrow(ax, (8, 3.4), (8, 2.85))
    draw_arrow(ax, (6, 6.1), (8, 4.6), connectionstyle="arc3,rad=-0.2")
    draw_arrow(ax, (7, 2.5), (6.5, 1.7), connectionstyle="arc3,rad=0.2")
    draw_arrow(ax, (8, 2.15), (7, 1.7), connectionstyle="arc3,rad=-0.2")
    draw_arrow(ax, (6, 0.9), (6, 0.65))
    
    # 수식 박스
    ax.text(10.5, 8.5, 'Key Insight:\n\nReal images:\n  SD U-Net predicts\n  noise accurately\n  → |ε - ε̂| small\n\nFake images:\n  Different distribution\n  → |ε - ε̂| large', 
            ha='left', va='top', fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import os
    
    # 저장 경로
    save_dir = "/home/deepfake/lju_workspace/Mymodel/result_png"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating DiffLaRE architecture diagrams...")
    
    # 1. 전체 아키텍처 (세로형)
    fig1 = draw_difflaare_architecture()
    fig1.savefig(f"{save_dir}/difflaare_architecture_full.png", dpi=300, bbox_inches='tight', 
                 facecolor='white', edgecolor='none')
    fig1.savefig(f"{save_dir}/difflaare_architecture_full.pdf", bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print(f"✓ Saved: {save_dir}/difflaare_architecture_full.png/pdf")
    
    # 2. 간소화된 가로형
    fig2 = draw_simplified_architecture()
    fig2.savefig(f"{save_dir}/difflaare_architecture_simple.png", dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    fig2.savefig(f"{save_dir}/difflaare_architecture_simple.pdf", bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print(f"✓ Saved: {save_dir}/difflaare_architecture_simple.png/pdf")
    
    # 3. LaRE 모듈 상세도
    fig3 = draw_lare_detail()
    fig3.savefig(f"{save_dir}/lare_module_detail.png", dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    fig3.savefig(f"{save_dir}/lare_module_detail.pdf", bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print(f"✓ Saved: {save_dir}/lare_module_detail.png/pdf")
    
    print("\n✅ All diagrams saved successfully!")
    print(f"📁 Location: {save_dir}/")
    
    # plt.show()  # 화면에 표시하려면 주석 해제
