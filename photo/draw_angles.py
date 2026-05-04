import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch

# 设置中文字体（避免乱码，英文标签可忽略）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 定义 MediaPipe 33关键点的简化坐标（站立姿态，正面）
# 索引对应官方编号，这里只取需要的关键点
keypoints = {
    # 上半身
    11: np.array([-0.3, 1.0]),   # 左肩
    12: np.array([0.3, 1.0]),    # 右肩
    13: np.array([-0.45, 0.75]), # 左肘
    14: np.array([0.45, 0.75]),  # 右肘
    15: np.array([-0.5, 0.5]),   # 左腕
    16: np.array([0.5, 0.5]),    # 右腕
    23: np.array([-0.25, 0.6]),  # 左髋
    24: np.array([0.25, 0.6]),   # 右髋
    25: np.array([-0.25, 0.25]), # 左膝
    26: np.array([0.25, 0.25]),  # 右膝
    27: np.array([-0.25, -0.1]), # 左踝
    28: np.array([0.25, -0.1]),  # 右踝
}

# 骨架连接关系（MediaPipe标准连接）
connections = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

fig, ax = plt.subplots(figsize=(6, 8))
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.3, 1.2)
ax.set_aspect('equal')
ax.axis('off')

# 绘制骨架连线
for (i, j) in connections:
    if i in keypoints and j in keypoints:
        x = [keypoints[i][0], keypoints[j][0]]
        y = [keypoints[i][1], keypoints[j][1]]
        ax.plot(x, y, 'gray', linewidth=2, alpha=0.7)

# 绘制关键点
for idx, pos in keypoints.items():
    ax.scatter(pos[0], pos[1], s=80, c='navy', zorder=5)
    ax.text(pos[0]+0.02, pos[1]+0.02, str(idx), fontsize=8, color='navy')

# 辅助函数：绘制角度弧线（给定三个点 A-B-C，在B处画弧）
def draw_angle_arc(ax, A, B, C, color, radius=0.08, label=''):
    BA = A - B
    BC = C - B
    angle_start = np.arctan2(BA[1], BA[0]) * 180 / np.pi
    angle_end = np.arctan2(BC[1], BC[0]) * 180 / np.pi
    # 确保逆时针方向
    if angle_start < 0:
        angle_start += 360
    if angle_end < 0:
        angle_end += 360
    # 计算最小角度差
    theta1 = min(angle_start, angle_end)
    theta2 = max(angle_start, angle_end)
    # 绘制弧线
    arc = Arc(B, width=2*radius, height=2*radius, angle=0,
              theta1=theta1, theta2=theta2, color=color, linewidth=2)
    ax.add_patch(arc)
    # 添加标签（弧线中点位置）
    mid_angle = (theta1 + theta2) / 2 * np.pi / 180
    label_pos = B + 1.5 * radius * np.array([np.cos(mid_angle), np.sin(mid_angle)])
    ax.text(label_pos[0], label_pos[1], label, fontsize=8, color=color, ha='center', va='center')

# 标注8个角度（左侧为例，右侧对称）
# 左肘角：肩(11) - 肘(13) - 腕(15)
draw_angle_arc(ax, keypoints[11], keypoints[13], keypoints[15], 'red', radius=0.1, label='Elbow')
# 左肩角：肘(13) - 肩(11) - 髋(23)
draw_angle_arc(ax, keypoints[13], keypoints[11], keypoints[23], 'blue', radius=0.12, label='Shoulder')
# 左髋角：肩(11) - 髋(23) - 膝(25)
draw_angle_arc(ax, keypoints[11], keypoints[23], keypoints[25], 'green', radius=0.1, label='Hip')
# 左膝角：髋(23) - 膝(25) - 踝(27)
draw_angle_arc(ax, keypoints[23], keypoints[25], keypoints[27], 'orange', radius=0.1, label='Knee')

# 右侧对称标注（右肘、右肩、右髋、右膝）
draw_angle_arc(ax, keypoints[12], keypoints[14], keypoints[16], 'red', radius=0.1, label='')
draw_angle_arc(ax, keypoints[14], keypoints[12], keypoints[24], 'blue', radius=0.12, label='')
draw_angle_arc(ax, keypoints[12], keypoints[24], keypoints[26], 'green', radius=0.1, label='')
draw_angle_arc(ax, keypoints[24], keypoints[26], keypoints[28], 'orange', radius=0.1, label='')

# 添加图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', lw=2, label='Elbow angle'),
    Line2D([0], [0], color='blue', lw=2, label='Shoulder angle'),
    Line2D([0], [0], color='green', lw=2, label='Hip angle'),
    Line2D([0], [0], color='orange', lw=2, label='Knee angle'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.title('Eight Joint Angles Extracted from MediaPipe Skeleton', fontsize=12, pad=20)
plt.tight_layout()
plt.savefig('angle_illustration.png', dpi=300, bbox_inches='tight')
plt.show()