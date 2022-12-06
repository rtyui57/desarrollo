import cv2

def get_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (1024, 1024))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_template():
    template = cv2.imread('media/template.png')
    #template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    template = cv2.resize(template, (1280,720))#/255.
    template_copy = template
    return template_copy