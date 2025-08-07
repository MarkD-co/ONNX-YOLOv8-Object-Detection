import cv2
import sys
import CoCo_data
import os

def add_track_draw(img, tracks, frame_count, file_log):


    class_color = {
        0 : (255,0,0),
        2 : (0,255,0),
        5 : (0,0,255),
        7 : (0,255,255)
    }

    for track in tracks:

        if not track.is_confirmed() or track.time_since_update > 0:
            continue  # 跳过未确认或已丢失的轨迹
    
        track_id = track.track_id
        class_id = int(track.get_det_class())
        class_name = CoCo_data.COCO_CLASSES[class_id]
        ltrb = track.to_ltrb()
        cla_color = (0,0,0)
        if (class_id == 0 or class_id == 2 or class_id == 7 or class_id == 5):
            cla_color = class_color[class_id]
        
        #解析新坐标
        x1, y1, x2, y2 = map(int, ltrb)



        #绘制方框
        cv2.rectangle(img, (x1, y1), (x2, y2), cla_color, 2)

        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
    
        label = f"{track_id}  {class_name}"

        # 在方框上方绘制文本背景
        text_size = cv2.getTextSize(label, fontFace, font_scale, thickness)[0]
        cv2.rectangle(img, 
                 (x1, y1 - text_size[1] - 4),
                 (x1 + text_size[0], y1),
                 cla_color, -1)  # 填充矩形
        

        #添加ID标签
        
        cv2.putText(img, text = label, org = (x1, y1 - 2), fontFace = fontFace, fontScale = 0.5, color = (255,255,255))

        #写入日志
        text = f"{frame_count}  {track_id}  {class_name}  {x1}, {y1}, {x2}, {y2}\n"
        file_log.write(text)


    return img






