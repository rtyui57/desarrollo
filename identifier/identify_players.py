import cv2
import numpy as np
from colorthief import ColorThief
import matplotlib.pyplot as plt

def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)


def get_centers(img):
    height, width, _ = np.shape(img)
    data = np.reshape(img, (height * width, 3))
    data = np.float32(data)
    number_clusters = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, number_clusters, None, criteria, 10, flags)
    return centers

def display(indice, centers, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bars = []
    rgb_values = []

    for index, row in enumerate(centers):
        bar, rgb = create_bar(200, 200, row)
        bars.append(bar)
        rgb_values.append(rgb)

    img_bar = np.hstack(bars)

    for index, row in enumerate(rgb_values):
        image = cv2.putText(img_bar, f'{index + 1}. RGB: {row}', (5 + 200 * index, 200 - 10),
                            font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        print(f'{index + 1}. RGB{row}')

    cv2.imshow('Image {}'.format(indice), img)
    cv2.imshow('Dominant colors {}'.format(indice), img_bar)

image = cv2.imread('media/players/hol3.jpg')
hvs = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
plt.imshow(hvs)
plt.show()

centers = get_centers(image)
display(1, centers, image)
""""
image2 = cv2.imread('media/players/arg2.jpg')
centers = get_centers(image2)
display(2, centers, image2)

ct = ColorThief('media/players/arg2.jpg')
palette = ct.get_palette(color_count=5)
plt.imshow([[palette[i] for i in range(5)]])
plt.show()

ct = ColorThief('media/players/arg1.jpg')
palette = ct.get_palette(color_count=5)
plt.imshow([[palette[i] for i in range(5)]])
plt.show()

ct = ColorThief('media/players/hol1.jpg')
palette = ct.get_palette(color_count=5)
plt.imshow([[palette[i] for i in range(5)]])
plt.show()

ct = ColorThief('media/players/hol2.jpg')
palette = ct.get_palette(color_count=5)
plt.imshow([[palette[i] for i in range(5)]])
plt.show()

ct = ColorThief('media/players/hol3.jpg')
palette = ct.get_palette(color_count=5)
plt.imshow([[palette[i] for i in range(5)]])
plt.show() """

""" image3 = cv2.imread('media/players/hol1.jpg')
centers = get_centers(image3)
display(3, centers, image3)

image4 = cv2.imread('media/players/hol2.jpg')
centers = get_centers(image4)
display(4, centers, image4) """

cv2.waitKey(0)