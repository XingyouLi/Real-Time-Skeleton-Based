import cv2
import mediapipe as mp
import numpy as np
import os

# 初始化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 数据路径
data_path = r"D:\action_recognition_project\date\raw"
save_root = r"D:\action_recognition_project\results"

# 九个类别
actions = [
    "WallPushups",
    "JumpingJack",
    "PushUps",
    "GolfSwing",
    "Lunges",
    "TennisSwing",
    "TaiChi",
    "PlayingGuitar",
    "BrushingTeeth"
]

for action in actions:
    print(f"\n===== 处理类别: {action} =====")

    action_path = os.path.join(data_path, action)
    save_path = os.path.join(save_root, action)

    os.makedirs(save_path, exist_ok=True)

    videos = os.listdir(action_path)

    for video in videos:
        video_file = os.path.join(action_path, video)

        # 只处理视频文件
        if not video_file.endswith((".avi", ".mp4")):
            continue

        print(f"处理视频: {video}")

        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print("❌ 打开失败:", video)
            continue

        all_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                frame_data = []

                for lm in landmarks:
                    frame_data.append([lm.x, lm.y, lm.z, lm.visibility])

                all_frames.append(frame_data)
            else:
                all_frames.append([[0, 0, 0, 0]] * 33)

        cap.release()

        # 转numpy
        data = np.array(all_frames)

        # 保存
        name = os.path.splitext(video)[0]
        npy_path = os.path.join(save_path, name + ".npy")

        np.save(npy_path, data)

        print(f"✅ 保存: {npy_path} 形状: {data.shape}")

print("\n🎉 全部处理完成！")