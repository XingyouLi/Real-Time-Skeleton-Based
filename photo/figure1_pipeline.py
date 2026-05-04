import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 颜色
c_input = '#f0f0f0'
c_pose = '#a6cee3'
c_angle = '#9ecae1'
c_delta = '#fdae61'
c_fusion = '#31a354'
c_lstm = '#de2d26'
c_out = '#f781bf'

def box(x, y, w, h, color, text_list, edge='black', lw=1.5):
    rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                  boxstyle="round,pad=0.1",
                                  facecolor=color, edgecolor=edge, linewidth=lw)
    ax.add_patch(rect)
    for i, (txt, size, bold, col) in enumerate(text_list):
        text_y = y + h/2 - 0.3 - i * 0.4
        ax.text(x, text_y, txt, ha='center', va='center',
                fontsize=size, fontweight='bold' if bold else 'normal', color=col)

def arrow(x1, y1, x2, y2, col='black', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw))

# 1. 输入视频
box(5, 9.0, 2.2, 1.0, c_input,
    [('Input Video Frames', 9, True, 'black'),
     ('(RGB, 30 fps)', 8, False, 'gray')])

arrow(5, 8.5, 5, 7.8)

# 2. MediaPipe 关键点提取
box(5, 7.3, 3.0, 1.2, c_pose,
    [('MediaPipe Pose', 10, True, 'black'),
     ('33 Keypoints per Frame', 8, False, 'black')])

arrow(5, 6.7, 5, 6.0)

# 分叉箭头
arrow(5, 6.0, 2.5, 5.2, 'black')
arrow(5, 6.0, 7.5, 5.2, 'black')

# 3. 角度特征（左分支）
box(2.5, 4.7, 2.8, 1.0, c_angle,
    [('Spatial Angle θ_t', 9, True, 'black'),
     ('8 joint angles', 8, False, 'black')])

# 4. 时序差分（右分支）
box(7.5, 4.7, 2.8, 1.0, c_delta,
    [('Temporal Diff Δθ_t', 9, True, 'black'),
     ('Δθ_t = θ_t - θ_{t-1}', 8, False, 'black')])

# 汇合箭头
arrow(2.5, 4.2, 5, 3.5, 'black')
arrow(7.5, 4.2, 5, 3.5, 'black')

# 5. 特征融合
box(5, 3.0, 3.0, 1.0, c_fusion,
    [('Fused Feature  X_t = [θ_t, Δθ_t]', 9, True, 'white'),
     ('16-dimensional vector', 8, False, 'white')],
    edge='black', lw=2)

arrow(5, 2.5, 5, 1.9)

# 6. LSTM 网络
box(5, 1.5, 3.0, 0.9, c_lstm,
    [('LSTM Network', 10, True, 'white'),
     ('2 Layers, 64 hidden units', 8, False, 'white')],
    edge='black', lw=2)

arrow(5, 1.05, 5, 0.5)

# 7. 输出
box(5, 0.2, 2.2, 0.7, c_out,
    [('Action Class', 9, True, 'black'),
     ('(9 categories)', 7, False, 'black')],
    edge='black')

# 步骤编号（左侧）
steps = [
    (5, 9.5, '①'),
    (5, 7.8, '②'),
    (2.5, 5.2, '③'),
    (7.5, 5.2, '④'),
    (5, 3.5, '⑤'),
    (5, 1.9, '⑥'),
    (5, 0.5, '⑦')
]
for x, y, num in steps:
    ax.text(x - 1.6, y, num, fontsize=11, fontweight='bold', color='gray')

ax.set_title('Figure 1. Overall framework of the proposed action recognition method.',
             fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figure1_pipeline.png', dpi=300, bbox_inches='tight')
plt.show()