import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_architecture_diagram(output_path="data/se_resnet8_architecture.png"):
    fig = plt.figure(figsize=(16, 7.8), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.8)
    ax.axis('off')

    # Color Palette - Professional Academic / Conference Style
    c_input_bg = '#EEF2FF'
    c_input_border = '#4F46E5'
    c_stem_bg = '#E0F2FE'
    c_stem_border = '#0284C7'
    c_res_bg = '#F8FAFC'
    c_res_border = '#334155'
    c_se_bg = '#FEF2F2'
    c_se_border = '#DC2626'
    c_policy_bg = '#ECFDF5'
    c_policy_border = '#059669'
    c_value_bg = '#FFFBEB'
    c_value_border = '#D97706'
    c_text_dark = '#0F172A'
    c_text_muted = '#475569'

    def draw_box(x, y, w, h, title, subtitle="", sub_lines=None, bg='#FFFFFF', border='#000000', radius=0.15, title_color='#0F172A', line_spacing=0.26):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f"round,pad={radius},rounding_size=0.12",
                             linewidth=1.8, edgecolor=border, facecolor=bg)
        ax.add_patch(box)
        
        # Title
        ax.text(x + w/2, y + h - 0.32, title, ha='center', va='center',
                fontsize=9.8, fontweight='bold', color=title_color, family='sans-serif')
        
        # Subtitle
        if subtitle:
            ax.text(x + w/2, y + h - 0.62, subtitle, ha='center', va='center',
                    fontsize=8.2, fontweight='bold', color=c_text_muted, family='sans-serif')
        
        # Detailed lines
        if sub_lines:
            curr_y = y + h - 0.96
            for line in sub_lines:
                if line == "":
                    curr_y -= line_spacing * 0.6
                    continue
                ax.text(x + w/2, curr_y, line, ha='center', va='center',
                        fontsize=7.6, color=c_text_dark, family='sans-serif')
                curr_y -= line_spacing

    def draw_arrow(x1, y1, x2, y2, label="", color='#64748B'):
        arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                        arrowstyle="Simple,tail_width=1.5,head_width=5.5,head_length=5",
                                        color=color)
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mid_x, mid_y + 0.16, label, ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', color='#475569', family='sans-serif',
                    bbox=dict(boxstyle="round,pad=0.15", fc='white', ec='#CBD5E1', lw=0.8, alpha=0.9))

    # Title Banner
    ax.text(8.0, 7.42, "SE-ResNet-8 Dual-Head Neural Chess Engine Architecture",
            ha='center', va='center', fontsize=14, fontweight='bold', color='#1E293B', family='sans-serif')
    ax.text(8.0, 7.08, "18-Plane Canonical Representation  •  4× Residual Blocks with Squeeze-and-Excitation  •  Decoupled Policy & Value Heads",
            ha='center', va='center', fontsize=8.8, color='#64748B', family='sans-serif')

    # 1. INPUT STAGE
    draw_box(0.5, 1.0, 2.3, 5.6, "1. Canonical Input", "FEN → 18 Planes",
             ["Board Representation:",
              "• 12 Piece Maps (P,N,B,R,Q,K)",
              "• 2 Legal Attack Maps (W/B)",
              "• 1 Pinned Piece Plane",
              "• 1 Check State Indicator",
              "• 2 Castling Rights (W/B)",
              "",
              "Shape: (Batch, 18, 8, 8)",
              "Canonical Color Inversion",
              "Active Player Perspective"],
             bg=c_input_bg, border=c_input_border, title_color='#3730A3', line_spacing=0.28)

    draw_arrow(2.8, 4.8, 3.4, 4.8, "18×8×8")

    # 2. STEM CONVOLUTION
    draw_box(3.4, 3.3, 2.2, 3.0, "2. Stem Conv", "Feature Lifting",
             ["Conv2d(18 → 128, 3×3)",
              "Stride = 1, Padding = 1",
              "BatchNorm2d(128)",
              "ReLU Activation",
              "",
              "Output: (B, 128, 8, 8)"],
             bg=c_stem_bg, border=c_stem_border, title_color='#0369A1', line_spacing=0.28)

    # 3. BACKBONE CONTAINER (4x SE-ResBlock)
    backbone_box = FancyBboxPatch((6.1, 0.75), 5.5, 5.85,
                                 boxstyle="round,pad=0.18,rounding_size=0.15",
                                 linewidth=1.8, edgecolor=c_res_border, facecolor=c_res_bg, linestyle='--')
    ax.add_patch(backbone_box)
    ax.text(8.85, 6.35, "3. Backbone: 4× SE-Residual Blocks (SE-ResNet-8)",
            ha='center', va='center', fontsize=10.2, fontweight='bold', color='#1E293B', family='sans-serif')
    ax.text(8.85, 6.06, "Identity Skip Connection  +  Squeeze-and-Excitation Channel Attention",
            ha='center', va='center', fontsize=7.8, color='#64748B', family='sans-serif')

    # Arrow from Stem to Backbone
    draw_arrow(5.6, 4.8, 7.1, 4.8, "128×8×8")

    # Inside Backbone: Detailed Layer Breakdown
    draw_box(7.1, 3.85, 4.2, 1.9, "Residual Convolutions", "Spatial Feature Extraction",
             ["Conv2d(128 → 128, 3×3) → BatchNorm → ReLU",
              "Conv2d(128 → 128, 3×3) → BatchNorm",
              "Output Features: (B, 128, 8, 8)"],
             bg='#FFFFFF', border='#64748B', title_color='#1E293B', line_spacing=0.27)

    draw_arrow(9.2, 3.85, 9.2, 3.35)

    draw_box(7.1, 0.95, 4.2, 2.4, "Squeeze-and-Excitation (SE)", "Channel Recalibration (r=8)",
             ["AdaptiveAvgPool2d(1×1) → Shape: (B, 128)",
              "Linear(128 → 16) + ReLU  [Squeeze]",
              "Linear(16 → 128) + Sigmoid [Excitation]",
              "Channel Weighting: X_scaled = X ⊙ Scale",
              "Residual Skip Add: Out = ReLU(X_scaled + Res)"],
             bg=c_se_bg, border=c_se_border, title_color='#B91C1C', line_spacing=0.27)

    # Identity Skip Connection inside Backbone
    # Branching point
    ax.plot(6.65, 4.8, marker='o', markersize=5, color='#475569')
    # Path: from (6.65, 4.8) -> down to (6.65, 1.35) -> right to (7.1, 1.35)
    ax.plot([6.65, 6.65], [4.8, 1.35], color='#475569', lw=1.8)
    skip_arrow = patches.FancyArrowPatch((6.65, 1.35), (7.1, 1.35),
                                        arrowstyle="Simple,tail_width=1.5,head_width=5.5,head_length=5",
                                        color="#475569")
    ax.add_patch(skip_arrow)
    ax.text(6.65, 3.1, "Identity\nSkip", ha='center', va='center',
            fontsize=7.2, fontweight='bold', color='#475569',
            bbox=dict(boxstyle="round,pad=0.18", fc='white', ec='#CBD5E1', lw=0.8, alpha=0.95))

    # Exit from Backbone to Decoupled Heads
    # Output path from Backbone at (11.6, 3.75)
    ax.plot([11.3, 11.8], [2.15, 2.15], color='#64748B', lw=1.8)
    ax.plot([11.8, 11.8], [2.15, 3.75], color='#64748B', lw=1.8)
    ax.plot([11.8, 12.0], [3.75, 3.75], color='#64748B', lw=1.8)
    ax.plot([12.0, 12.0], [5.25, 2.4], color='#64748B', lw=1.8)
    draw_arrow(12.0, 5.25, 12.4, 5.25, "128×8×8")
    draw_arrow(12.0, 2.4, 12.4, 2.4, "128×8×8")

    # 4A. POLICY HEAD
    draw_box(12.4, 3.9, 3.3, 2.7, "4A. Policy Head", "Tactical Move Prediction",
             ["Conv2d(128 → 64, 1×1) + BN + ReLU",
              "Flatten → 64 × 8 × 8 = 4,096 dims",
              "Linear(4096 → 4096)",
              "",
              "Output: 4,096 Discrete Move Logits",
              "(Categorical Cross-Entropy Loss)"],
             bg=c_policy_bg, border=c_policy_border, title_color='#047857', line_spacing=0.27)

    # 4B. VALUE HEAD
    draw_box(12.4, 1.05, 3.3, 2.7, "4B. Value Head", "Position Evaluation Certainty",
             ["Conv2d(128 → 32, 1×1) + BN + ReLU",
              "Flatten → 32 × 8 × 8 = 2,048 dims",
              "Linear(2048 → 128) + ReLU",
              "Linear(128 → 1) + Tanh",
              "Output: Scalar V(s) ∈ [-1.0, +1.0]",
              "(MSE Loss with Anti-Delusion Clamp)"],
             bg=c_value_bg, border=c_value_border, title_color='#B45309', line_spacing=0.27)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Architecture diagram successfully saved to {output_path}")

if __name__ == "__main__":
    create_architecture_diagram()
