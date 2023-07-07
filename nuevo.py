import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from flask import Flask, render_template, Response

app = Flask(__name__)

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

classes_to_filter = ['1'] #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]


opt  = {
    
    "weights": "weights/last.pt", # Path to weights file default weights are for nano model
    "yaml"   : "data/coco.yaml",
    "img-size": 416, # default image size
    "conf-thres": 0.25, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter  # list of classes to filter or None

}
camara = cv2.VideoCapture("https://isaopen.ezvizlife.com/v3/openlive/G85200741_1_2.m3u8?expire=1750890528&id=597313050621853696&c=cf2d0e104f&t=db6fd0caef82abad804c303737021495f33baf86edf6384960b5f745033cf70e&ev=100")

# para acceder a cámara predeterminada
#camara = cv2.VideoCapture(1)  # Índice 0 para acceder a la cámara predeterminada



# Obtener información del video
fps = camara.get(cv2.CAP_PROP_FPS)
w = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = 0  # La transmisión en vivo no tiene un número fijo de cuadros

# Inicializar el objeto para escribir el video de salida
output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
torch.cuda.empty_cache()

# Inicializar el modelo y configurarlo para inferencia
with torch.no_grad():
    weights, imgsz = opt['weights'], opt['img-size']
    set_logging()
    device = select_device(opt['device'])
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)  # Cargar el modelo FP32
    stride = int(model.stride.max())  # Stride del modelo
    imgsz = check_img_size(imgsz, s=stride)  # Verificar el tamaño de imagen
    if half:
        model.half()

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    classes = None
    if opt['classes']:
        classes = []
        for class_name in opt['classes']:
            classes.append(names.index(class_name))

    if classes:
        classes = [i for i in range(len(names)) if i not in classes]

    def gen_frames():
        while True:
            ret, img0 = camara.read()
            if not ret:
                break

            img = letterbox(img0, imgsz, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR a RGB, a 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 a fp16/32
            img /= 255.0  # 0 - 255 a 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inferencia
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
            t2 = time_synchronized()

            for i, det in enumerate(pred):
                s = ''
                s += '%gx%g ' % img.shape[2:]  # Cadena de impresión
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # Detecciones por clase
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # Agregar a la cadena

                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

            output.write(img0)
            frame = cv2.imencode('.jpg', img0)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Transmitir cuadro como respuesta

    @app.route('/')
    def index():
        return render_template('./index.html')

    @app.route('/video_feed')
    def video_feed():
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
