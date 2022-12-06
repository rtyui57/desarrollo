import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from matplotlib import pyplot as plt

COLORS =  {
    0 : (0, 222, 255),
    1 : (255, 171, 0),
    2 : (0, 0, 0),
    3 : (230, 255, 0),
    4 : (12, 96, 117)
}

def remove_background(imagen):
    hsv = cv2.cvtColor(np.copy(imagen), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    imask = mask==0
    m = imask.astype("uint8")
    target = cv2.bitwise_and(imagen, imagen, mask=m)
    return target

def quantize(img, clusters):
    dominant_color = (0, 0, 0)
    h, w = img.shape[:2]
    image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clf = MiniBatchKMeans(n_clusters=clusters)
    labels = clf.fit_predict(image)
    centers =  clf.cluster_centers_.astype("uint8")
    colors = cv2.cvtColor(np.array([centers]), cv2.COLOR_LAB2RGB)[0]
    for color in colors:
        if color[0] > 3 and color[1] > 3 and color[2] > 3:
            dominant_color = color
    #cs, rep = np.unique(labels, return_counts=True)
    quantized = centers[labels]
    quantized = quantized.reshape((h, w, 3))
    quantized = cv2.cvtColor(quantized, cv2.COLOR_LAB2RGB, cv2.CV_8U)
    return quantized, dominant_color

def classify(players, puntos):
    colors = []
    for player in players:
        jug = np.copy(player)
        jug = remove_background(jug)
        jug = jug[int(jug.shape[0]*0.05):int(jug.shape[0]*0.6), :]
        quantized, dom_color = quantize(jug, clusters=2)
        colors.append(dom_color)
    clf = DBSCAN(eps=30, min_samples=0)
    labels = clf.fit_predict(colors)
    for i in range(0, len(labels)):
        clase = labels[i]
        puntos[i].append(COLORS[clase])