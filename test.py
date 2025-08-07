import argparse
import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO
import onnx

model = YOLO("yolov8m.pt")
model.export(format="onnx")
