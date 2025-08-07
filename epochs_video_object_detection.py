import os
import video_object_detection

#输入视频文件夹
input_dir_path = "input_media"
dir_list = os.listdir(input_dir_path)
doc_num = len(dir_list)
doc_count = 0
for file_name in dir_list:
    name, ext = os.path.splitext(file_name)
    file_path = os.path.join(input_dir_path, file_name)

    video_object_detection.single_video_object_detection(file_path, name)
    doc_count += 1

    print(f"-----当前处理进度：{doc_count}/{doc_num}")



