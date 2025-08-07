import cv2
import os
from yolov8 import YOLOv8
import time

#总计时_0
all_start = time.perf_counter()

# 初始化yolov8目标检测器
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

# 确保输出目录存在
os.makedirs("output_image", exist_ok=True)


#批量处理

##确认图片所在文件夹
images_doc = "input_image"

# 读取图像
image_num = 0
image_total = len(os.listdir(images_doc))
print("待处理图片总计：{}".format(image_total))
for filename in os.listdir(images_doc):

    #计时_0
    time_start = time.perf_counter()

    img_path = os.path.join(images_doc,filename)
    img = cv2.imread(img_path)

    # 检查图片是否正确读取
    if img is None:
        print(f"错误：无法读取图片 {img_path}")
        continue

    image_num += 1

    # 检测物体
    boxes, scores, class_ids = yolov8_detector(img)

    # 绘制检测结果并保存
    combined_img = yolov8_detector.draw_detections(img)
    output_path = os.path.join("output_image",filename)
    cv2.imwrite(output_path, combined_img)

    #计时_1
    time_end = time.perf_counter()

    print(f"检测结果已保存至: {output_path} ({image_num}/{image_total}), 用时: {round(time_end - time_start, 2)}s")

#总计时_1
al_end = time.perf_counter()
al_time_pre = round(al_end - all_start, 2)
al_time = round(al_end - all_start)

print("共计{}张图片已处理,总计用时: {}min{}s".format(image_num, al_time/60, al_time%60 + (al_time_pre - al_time)))