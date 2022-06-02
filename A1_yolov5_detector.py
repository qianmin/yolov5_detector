import cv2
import argparse
import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import copy


def create_random_color():
    # 功能：产生随机RGB颜色
    # 输出：color <class 'tuple'> 颜色

    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    color = (r, g, b)

    return color


def draw_result(im, li_xyxy, li_label):
    if (len(li_xyxy)):
        for i in range(0, len(li_xyxy)):
            # im = draw_one_box(im, li_xyxy[i], li_label[i])
            im = draw_one_box_chinese(im, li_xyxy[i], li_label[i])
    return im


class Yolov5Detector():
    def __init__(self):
        # Initialize
        self.device = select_device()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        # Load model
        device = ''
        self.device = select_device(device)
        weights = "./yolov5s.pt"
        dnn = False
        data = './data/coco128.yaml'  # dataset.yaml path

        half = False
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half)
        stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = (640, 640)
        imgsz = check_img_size(imgsz, s=stride)  # check image size

    def run(self, img0, conf_thres=0.25):
        # img prepare
        # Padded resize
        img_size = 640
        stride = 32
        auto = True
        img = letterbox(img0, img_size, stride=stride, auto=auto)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        dt, seen = [0.0, 0.0, 0.0], 0
        ######
        im = img
        im0s = img0
        ######
        im = torch.from_numpy(im).to(self.device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        pred = self.model(im)
        t3 = time_sync()

        # NMS
        iou_thres = 0.45
        classes = [0, 1, 2, 3, 4, 5, 6, 7]
        agnostic_nms = False
        max_det = 1000
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0 = im0s.copy()
            # s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0  # for save_crop
            line_thickness = 3
            annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                li_xyxy = []
                li_label = []
                for *xyxy, conf, cls in reversed(det):
                    view_img = True
                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{self.names[c]} {conf:.2f}'
                        li_xyxy.append(xyxy)
                        li_label.append(label)

                        # annotator.box_label(xyxy, label, color=colors(c, True))
                return li_xyxy, li_label
            else:
                return [], []



def qmpltimg(cvimg, size=3):
    img_show = copy.deepcopy(cvimg)
    plt.figure(dpi=300, figsize=(size, 3))
    img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    plt.imshow(img_show)


def draw_one_box_only(im, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    #
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    #
    p3 = (int(box[0]), int((box[1] + box[3]) / 2))
    p4 = (int(box[2]), int((box[1] + box[3]) / 2))
    p33 = (p3[0] + int(0.1 * (p4[0] - p3[0])), p3[1])
    p44 = (p4[0] - int(0.1 * (p4[0] - p3[0])), p4[1])
    cv2.line(im, p33, p44, color=(255,0,0), thickness=lw, lineType=cv2.LINE_AA)

    p3 = (int((box[0] + box[2]) / 2), int(box[1]))
    p4 = (int((box[0] + box[2]) / 2), int(box[3]))
    p33 = (p3[0], p3[1] + int(0.1 * (p4[1] - p3[1])))
    p44 = (p4[0], p4[1] - int(0.1 * (p4[1] - p3[1])))
    cv2.line(im, p33, p44, color=(255,0,0), thickness=lw, lineType=cv2.LINE_AA)

    pcx=int((p1[0]+p2[0])/2)
    pcy=int((p1[1]+p2[1])/2)
    r=int((0.4 * (p4[1] - p3[1]))/2)
    cv2.circle(im,(pcx,pcy),r,color=(255,0,0), thickness=lw, lineType=cv2.LINE_AA)

    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(im,
        #             label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
        #             0,
        #             lw / 3,
        #             txt_color,
        #             thickness=tf,
        #             lineType=cv2.LINE_AA)
    return im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), int(lw / 3)
    # return im,label,(p1[0], p1[1]+2 if outside else p1[1] + h + 2),int(lw/3)


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_result(im, li_xyxy, li_label):
    if (len(li_xyxy)):
        li_c = []
        for i in range(0, len(li_xyxy)):
            im, label, pos, size = draw_one_box_only(im, li_xyxy[i], li_label[i])
            li_c.append([label, pos, size])
            # print([label, pos, size])

        for i in range(len(li_c)):
            if ("person" in li_c[i][0]):
                label = "人"
            elif ("car" in li_c[i][0]):
                label = "车"
            else:
                label="dont care"
            im = cv2AddChineseText(img=im, text=label, position=li_c[i][1], textColor=(0, 255, 0), textSize=30)
    return im


# img = draw_result(frame, li_xyxy, li_label)
# qmpltimg(img)
# qmpltimg(frame)

if __name__ == '__main__':
    # cap=cv2.VideoCapture("./test_img/dance2.mp4")
    cap = cv2.VideoCapture(0)
    net=Yolov5Detector()
    while(1):
        ret,frame=cap.read()
        t1=time.time()
        li_xyxy,li_label=net.run(frame)

        img=draw_result(frame,li_xyxy,li_label)
        cv2.imshow("",img)
        cv2.waitKey(33)
        print("time:",str(time.time()-t1)[:6])

