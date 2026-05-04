import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

color_raw = '#d9d9d9'
color_angle = '#9ecae1'
color_delta = '#fdae61'
color_fusion = '#31a354'
color_lstm = '#de2d26'

def box(x, y, w, h, color, text_list, edge='black', lw=1.5):
    rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                  boxstyle="round,pad=0.1",
                                  facecolor=color, edgecolor=edge, linewidth=lw)
    ax.add_patch(rect)
    for i, (txt, size, bold, col) in enumerate(text_list):
        text_y = y + h/2 - 0.3 - i * 0.4
        ax.text(x, text_y, txt, ha='center', va='center',
                fontsize=size, fontweight='bold' if bold else 'normal', color=col)

def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# 第一行：原始关键点
box(5, 8.5, 3.0, 0.8, color_raw,
    [('33 Keypoints × 3 (x, y, z)', 10, True, 'black'),
     ('99-dimensional raw vector', 8, False, 'gray')])
arrow(5, 8.1, 5, 7.3)

# 第二行：角度计算
box(5, 6.9, 2.0, 0.5, 'white', [('Angle Computation', 9, False, 'gray')],
    edge='gray', lw=1)
arrow(2.5, 6.6, 2.5, 6.0)
arrow(7.5, 6.6, 7.5, 6.0)

# 第三行左右
box(2.5, 5.5, 2.8, 0.7, color_angle,
    [('Spatial Angle θ_t', 9, True, 'black'),
     ('8-dimensional', 8, False, 'black')])
box(7.5, 5.5, 2.8, 0.7, color_delta,
    [('Temporal Diff Δθ_t', 9, True, 'black'),
     ('8-dimensional', 8, False, 'black')])
arrow(2.5, 5.15, 5, 4.3)
arrow(7.5, 5.15, 5, 4.3)

# 第四行：融合
box(5, 3.8, 3.0, 0.7, color_fusion, 
    [('Fused Feature X_t = [θ_t, Δθ_t]', 9, True, 'white'),
     ('16-dimensional', 8, False, 'white')],
    edge='black', lw=2)
arrow(5, 3.45, 5, 2.8)

# 第五行：LSTM
box(5, 2.3, 2.5, 0.8, color_lstm,
    [('LSTM Network', 10, True, 'white'),
     ('Sequence \u2192 Class Probabilities', 8, False, 'white')],
    edge='black', lw=2)
arrow(5, 1.9, 5, 1.3)

# 第六行：输出
box(5, 0.8, 2.5, 0.7, 'white',
    [('Action Classes (9 categories)', 9, True, 'black')],
    edge='black', lw=1.5)

ax.set_title('Feature Fusion Pipeline: From Keypoints to Classification', 
             fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('feature_fusion_pipeline.png', dpi=300, bbox_inches='tight')
plt.show()