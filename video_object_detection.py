import cv2
import time
import os
import sys
from yolov8 import YOLOv8  # 假设这是您的YOLOv8实现模块
from deep_sort_realtime.deepsort_tracker import DeepSort
from add_track_draw import add_track_draw

def single_video_object_detection(file_path, name):
    # 禁用OpenCV的GUI功能（解决无GUI环境问题）
    os.environ['OPENCV_IO_ENABLE_OPENCL'] = '0'
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

    # 设置输入输出文件路径
    input_video = file_path  # 替换为您的本地视频文件路径
    output_video = "output_media/" + name + ".mp4"  # 输出视频文件路径

    # 检查输入文件是否存在
    if not os.path.exists(input_video):
        print(f"错误：输入视频文件不存在: {input_video}")
        sys.exit(1)

    # 初始化视频捕获对象
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {input_video}")
        sys.exit(1)

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {frame_width}x{frame_height} @ {fps:.2f} FPS, 总帧数: {total_frames}")

    # 设置起始时间 (可选)
    start_seconds = 0  # 跳过前5秒
    start_frame = int(start_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 初始化视频写入对象 - 使用更兼容的编码
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4格式
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # 初始化YOLOv8模型 - 仅使用CPU执行提供程序
    print("初始化YOLOv8模型...")
    model_path = "models/yolov8m.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.4, iou_thres=0.5)

    print("初始化DeepSORT追踪器...")
    tracker = DeepSort(
        max_age=20,              # 减少漂移框保留时间
        n_init=3,                # 更快确认新轨迹
        nms_max_overlap=1.0,
        max_cosine_distance=0.35,  # 放宽外观匹配
        nn_budget=100,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=False
    )

    # 性能计数器
    frame_count = start_frame
    start_time = time.time()
    last_log_time = start_time
    processed_frames = 0

    #日志路径
    log_name = name + ".txt"
    log_path = os.path.join("logs", log_name)
    os.makedirs(os.path.dirname(log_path), exist_ok= True)
    file_log = open(log_path, "w", encoding = "utf-8")
    file_log.write("----------追踪目标日志----------\n")
    file_log.write("以下为各列数值含义：\n")
    file_log.write("帧数  目标ID  目标类别  x1, y1, x2, y2 (方框左上角以及右下角)\n")


    print(f"开始处理视频: {input_video}")
    print("按Ctrl+C可提前终止处理...")

    try:
        while cap.isOpened():
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break  # 视频结束或读取失败
            
            frame_count += 1
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # 执行目标检测
            boxes, scores, class_ids = yolov8_detector(frame)

            
            # 绘制检测结果
            #combined_img = yolov8_detector.draw_detections(frame)
            
            # 添加帧号信息
            cv2.putText(frame, f"Frame: {current_frame}/{total_frames}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ##Deepsort
            deepsort_detection = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                w = x2 - x1 
                h = y2 - y1

                deepsort_detection.append(([x1, y1, w, h], scores[i], str(class_ids[i])))

            tracks = tracker.update_tracks(deepsort_detection, frame = frame)
            track_img = add_track_draw(frame, tracks, frame_count, file_log)
            
            # 保存处理后的帧
            out.write(track_img)
            processed_frames += 1
            
            # 每5秒或每100帧打印一次进度
            current_time = time.time()
            if current_time - last_log_time > 5 or processed_frames % 100 == 0:
                elapsed = current_time - start_time
                if elapsed > 0:
                    fps = processed_frames / elapsed
                    progress = current_frame / total_frames * 100
                    remaining_time = (total_frames - current_frame) / fps if fps > 0 else 0
                    
                    print(f"进度: {progress:.1f}% | "
                        f"已处理: {processed_frames}/{total_frames}帧 | "
                        f"速度: {fps:.1f}FPS | "
                        f"已用: {elapsed:.0f}秒 | "
                        f"剩余: {remaining_time:.0f}秒")
                    last_log_time = current_time

    except KeyboardInterrupt:
        print("\n用户中断处理...")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
    finally:
        # 释放资源
        cap.release()
        out.release()
        file_log.close()
        
        # 计算最终性能
        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = processed_frames / total_time if total_time > 0 else 0

        print("\n处理完成!")
        print(f"处理帧数: {processed_frames}/{total_frames}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均FPS: {avg_fps:.2f}")
        print(f"输出视频已保存至: {output_video}")


if __name__ == "__main__":
    file_path = "input_media/.mp4"
    name = "input"
    single_video_object_detection(file_path, name)