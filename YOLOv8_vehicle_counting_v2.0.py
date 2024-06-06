import cv2
import time
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8x.pt')  # 加载模型

cap = cv2.VideoCapture('YOLO_test_video_02.mp4')  # 识别视频源

prev_time = time.time()  # 帧计数器

# 定义识别线线的起点和终点坐标
line_start = np.array([250, 450])  # 起点
line_end = np.array([850, 450])  # 终点

counter = 0  # 计数器
counter_id = 0  # 分配唯一 ID 的计数器
buffer_tracks = {}  # 保存位置和时间的元组
crossed_boxes = {}  # 用于存储每个框的交叉状态的字典
identification_range = 30  # 识别范围

max_track_length = 35  # 最大轨迹长度
track_timeout = 10  # 轨迹超时时间（秒）

unupdated_timeout = 0.1  # 未更新的轨迹存在时间（秒）
last_updated = {}  # 初始化 last_updated 字典


def main():
    global prev_time, counter, counter_id
    while True:
        ret, frame = cap.read()
        if not ret:
            # 单次播放
            # break

            # 循环播放
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue


        # 推理部分
        results = model.predict(frame, save=False, imgsz=320, conf=0.4, device="cuda:0", half=True, classes=[2, 7])

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            # 计算 FPS
            curr_time = time.time()
            FPS = 1 / (curr_time - prev_time)
            prev_time = curr_time
            FPS_str = 'FPS: %s' % round(FPS, 2)
            # 绘制边界框和 FPS
            frame_ = results[0].plot()
            cv2.putText(frame_, FPS_str, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (34, 139, 34), 2)

            for box in boxes:
                center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=int)

                cv2.circle(frame_, tuple(center), radius=5, color=(0, 255, 255), thickness=-1)

                cv2.line(frame_, tuple(line_start), tuple(line_end), color=(255, 0, 255), thickness=6)

                # 检查框 ID 是否已存在于缓冲区中
                box_id = None
                for id, past_positions in buffer_tracks.items():
                    if len(past_positions) > 0 and np.linalg.norm(
                            past_positions[-1][0] - center) < identification_range:  # 如果 y 位置接近缓冲区中的 1
                        box_id = id
                        break
                '''
                这段代码是用于判断当前的识别框（box）是否在之前的帧中已经识别过。
                在这段代码中，buffer_y_positions是一个字典，它的键是识别框的ID，值是该识别框在过去的帧中所有的y坐标位置。
                这段代码的目标是尝试找到一个已经存在于buffer_y_positions的识别框ID，使得该识别框在上一帧的y坐标位置与当前识别框的y坐标位置非常接近（差距小于30像素）。如果找到了这样的ID，则认为当前的识别框在之前的帧中已经识别过，于是将box_id设置为这个ID，并跳出循环。
                这种做法的目的是尽可能地追踪每一个识别框的运动，即使在连续的帧中，由于摄像设备的振动或者物体的移动速度等原因，相同的物体可能被检测出稍微不同的识别框。通过将新的识别框和旧的识别框相关联，我们可以对物体的运动进行追踪，例如判断物体何时穿过了一条特定的线。
                '''

                if box_id is None:
                    # 为盒子分配一个新的 ID
                    box_id = counter_id
                    counter_id += 1
                    buffer_tracks[box_id] = []
                    crossed_boxes[box_id] = False  # 将交叉状态初始化为 False

                buffer_tracks[box_id].append((center, time.time()))  # 添加位置和当前时间
                last_updated[box_id] = time.time()  # 更新时间

                # 检查框是否越界
                if len(buffer_tracks[box_id]) > 1:
                    if (buffer_tracks[box_id][-2][0][1] - line_start[1]) * (
                            buffer_tracks[box_id][-1][0][1] - line_start[1]) < 0:
                        # 盒子已经越界了
                        crossed_boxes[box_id] = True  # 更新交叉状态为 True
                        counter += 1

                # 如果框被标记为已越线，则绘制文本
                if crossed_boxes[box_id]:
                    cv2.putText(frame_, 'crossed', (center[0], center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

        for id, track in list(buffer_tracks.items()):
            if len(track) > max_track_length:  # 如果轨迹过长
                track.pop(0)  # 移除最旧的位置
            if time.time() - track[-1][1] > track_timeout:  # 如果轨迹过期
                del buffer_tracks[id]  # 删除整个轨迹

        for id in list(buffer_tracks.keys()):
            if time.time() - last_updated[id] > unupdated_timeout:  # 如果轨迹过期
                del buffer_tracks[id]  # 删除整个轨迹
                del last_updated[id]  # 删除对应的时间



        for track in buffer_tracks.values():
            positions = [pos_time[0] for pos_time in track]  # 提取位置列表
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    start_pos = positions[i - 1]
                    end_pos = positions[i]
                    cv2.line(frame_, start_pos, end_pos, (255, 255, 0), 4)  # 轨迹线条

        counter_str = 'Counter: %s' % counter
        cv2.putText(frame_, counter_str, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (34, 139, 34), 2)  # 将计数器放在框架上

        # 显示带有边界框的图像
        cv2.imshow('Live Detections', frame_)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    main()
    cap.release()
    cv2.destroyAllWindows()