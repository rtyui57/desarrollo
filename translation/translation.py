import cv2
from identifier.classifier import classify

def sum(x, y, array):
    return x*array[0] + y*array[1] + array[2]

def compute(x, y, H):
    return (sum(x, y, H[0])/sum(x,y,H[2])), (sum(x, y, H[1])/sum(x,y,H[2]))

def get_players(results, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    coord = results.xyxy[0]
    detecciones = coord.size(dim=0)
    puntos = []
    players = []
    for obj in range(0, detecciones):
        x1 = int(coord[obj][0].item())
        y1 = int(coord[obj][1].item())
        x2 = int(coord[obj][2].item()) 
        y2 = int(coord[obj][3].item())
        puntos.append([int(((x1 + x2)/2)), y2])
        player = image[y1:y2, x1:x2]
        players.append(player)
        writen = cv2.imwrite("media\\players\\{}.jpg".format(obj), player)
        if not writen:
            print("Not writen")
    classify(players, puntos)
    return puntos, players

def pintar_puntos(img, puntos):
    for punto in puntos:
        x = int(punto[0])
        y = int(punto[1])
        color = punto[2]
        print(color)
        img = cv2.circle(img, (x, y), 3, color, -1)
    return img

def draw_template(template, puntos, H):
    dst_points = []
    for punto in puntos:
        x, y = compute(punto[0], punto[1], H)
        dst_points.append([int(x), int(y), punto[2]])
    return pintar_puntos(template, dst_points)