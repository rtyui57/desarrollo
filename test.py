import torch
import numpy as np
import cv2
import tensorflow as tf
from modelos.keypoint import KeypointDetectorModel
from common.homography import visualize, _points_from_mask, denormalize
from common.translation import get_players, draw_template
from matplotlib import pyplot as plt

def get_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (1024, 1024))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_template():
    template = cv2.imread('media/template.png')
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    template = cv2.resize(template, (1280,720))/255.
    template_copy = template
    return template_copy

def load_keypoint():
    kp_model = KeypointDetectorModel(backbone='efficientnetb3', num_classes=29, input_shape=(512, 512),)
    WEIGHTS_PATH = ("models/keypoint_detector.h5")
    WEIGHTS_NAME = "keypoint_detector.h5"
    WEIGHTS_TOTAR = False
    checkpoints = tf.keras.utils.get_file(WEIGHTS_NAME, WEIGHTS_PATH, WEIGHTS_TOTAR,)
    kp_model.load_weights(checkpoints)
    return kp_model

def get_matrix(kp_model, imagen):
    pr_mask = kp_model(imagen)
    visualize(image=denormalize(imagen.squeeze()),pr_mask=pr_mask[..., -1].squeeze(),)
    src,dst = _points_from_mask(pr_mask[0])
    M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5)
    return M

def show_template(model, template, Matrix):
    detecciones = model(cv2.resize(image, (512, 512)))
    puntos = get_players(detecciones)
    temp = draw_template(cv2.resize(template, (320, 320)), puntos, Matrix)
    return temp

yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')
image = get_image('media/test_image.jpg')
template = get_template()
kp_model = load_keypoint()
M = get_matrix(kp_model, image)
template_2d = show_template(yolo, template, M)
plt.imshow(template_2d)
plt.show()
cv2.COLOR_RGB2