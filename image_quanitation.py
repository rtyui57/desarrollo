import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot as plt
from colorthief import ColorThief
from mpl_toolkits import mplot3d

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

paths = ['media/players/arg1.jpg', 'media/players/arg2.jpg', 'media/players/arg3.jpg', 'media/players/arg4.jpg', 'media/players/arg5.jpg',
        'media/players/hol1.jpg', 'media/players/hol2.jpg', 'media/players/hol3.jpg', 'media/players/hol4.jpg', 'media/players/hol5.jpg']
clusters = 2
path = 'media/players/{}.jpg'
dest = "media\\players\\quantized1.jpg"

colors = []
r = []
g = []
b = []
for i in range(0, 13):
    if i == 2 or i == 3:
        continue
    image = cv2.imread(path.format(i))
    image = remove_background(image)
    image = image[int(image.shape[0]*0.05):int(image.shape[0]*0.6), :]
    quantized, dom_color = quantize(image, clusters=clusters)
    colors.append(dom_color)
    plt.figure("Numeo jugaor {}".format(i))
    plt.imshow(quantized)
    plt.show()
    r.append(dom_color[0])
    g.append(dom_color[1])
    b.append(dom_color[2])

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.scatter(r, g, b, r)
ax.set_title('3D line plot geeks for geeks')
plt.show()
#cv2.imshow("a", np.hstack([image, quantized]))
cv2.imwrite(dest, quantized)
i =  cv2.cvtColor(cv2.imread(dest), cv2.COLOR_BGR2RGB)
print(colors)
clf = MiniBatchKMeans(n_clusters=2)
labels = clf.fit_predict(colors)
centers =  clf.cluster_centers_.astype("uint8")
print(labels, centers)
plt.imshow(quantized)
plt.show()
#cv2.waitKey(0)