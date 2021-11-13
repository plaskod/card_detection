import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os

try:
    from cv2 import cv2 as cv
except ImportError:
    pass
from skimage import filters

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)


""" PRZETWARZANIE OBRAZU """
def erosion_dilation(x, n, kernel_size = 3, erode_iterations = 1, dilate_iterations = 1):
    matrix_of_ones = np.ones((kernel_size, kernel_size), np.uint8)
    for _ in range(n):
        x = cv.erode(x, matrix_of_ones, iterations = erode_iterations)
        x = cv.dilate(x, matrix_of_ones, iterations = dilate_iterations)
    return x

def dilation_erosion(x, n, kernel_size = 3, erode_iterations = 1, dilate_iterations = 1):
    matrix_of_ones = np.ones((kernel_size, kernel_size), np.uint8)
    for _ in range(n):
        x = cv.dilate(x, matrix_of_ones, iterations = dilate_iterations)
        x = cv.erode(x, matrix_of_ones, iterations = erode_iterations)
    return x

""" PRZETWARZANIE KART """
def card_rank(id_card_rank):
    return {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "Walet": 11,
        "Dama": 12,
        "Król": 13,
        "As": 14,
    }[id_card_rank]

def card_suit(id_card_suit):
    return{
        "Trefl": 1,
        "Karo": 2,
        "Kier": 3,
        "Pik": 4,
    }[id_card_suit]

class Card:
    def __init__(self):
        self.contour = []


def show(image):
    cv.imshow("Karta", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def card_contouring(filepath):
    img = cv.imread(filepath)
    global WIDTH, HEIGHT
    dim = (WIDTH, HEIGHT)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # imgray = cv.GaussianBlur(imgray, (3,3), 0, borderType = cv2.BORDER_CONSTANT)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    thresh = erosion_dilation(thresh, 10)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print(contours)
    print('---')
    print(hierarchy)
    # cv.putText(img, "jakis tekst", (500,375), cv.FONT_HERSHEY_COMPLEX, 1.0, WHITE, 2)
    
    print(len(contours)) # number of detected countours, ~ ideally number of detected cards
    cv.drawContours(img, contours, -1, GREEN ,3)
    show(img) # TODO: switch to matplotlib for showing multiple images
    return img

def main(path):
    images = OrderedDict({})
    for (root, _, filenames) in os.walk(path):
        for filename in filenames:
            images[filename] = card_contouring(root+filename)

    
    """ Ponizszy fragment laczy i wyswietla obrazki horyzontalnie"""
    # tmp = np.hstack((img, img))
    # cv.imshow("Obrazki na horyzoncie", tmp)
    for imagename in images.keys():
        show(images[imagename])

    # TODO: Dipslay multiple images https://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/

def process_selected_images(path, selected_images):
    for filename in selected_images:
        card_contouring(path+filename)

if __name__ == '__main__':
    WIDTH = 1600
    HEIGHT = 900
    PATH = './karty/'
    selected_images = ['dwie_pary.jpg','krol_karo.jpg']

    process_selected_images(PATH, selected_images)
