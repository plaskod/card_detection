"""
Card detection

references:
[1] C. Zheng, R. Green, ‘Playing Card Recognition Using Rotational Invariant Template Matching’, Proceedings of
Image and Vision Computing New Zealand 2007, pp. 276-281, Hamilton, New Zealand, December 2007.
[2] J. Pimentel and A. Bernardino, “A Comparison of Methods for Detection and Recognition of Playing Cards.”
[3] Martins P., Reis L.P., Teófilo L. (2011) Poker Vision: Playing Cards and Chips Identification Based on Image
Processing. In: Vitrià J., Sanches J.M., Hernández M. (eds) Pattern Recognition and Image Analysis. IbPRIA 2011.
Lecture Notes in Computer Science, vol 6669. Springer, Berlin, Heidelberg
[4] D. Ballard, “Generalizing the Hough transform to detect arbitrary shapes,” Pattern Recognition, vol. 13, no. 2,
pp. 111–122, 1981.

"""

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


def erosion_dilation(x, n, kernel_size=3, erode_iterations=1, dilate_iterations=1):
    matrix_of_ones = np.ones((kernel_size, kernel_size), np.uint8)
    for _ in range(n):
        x = cv.erode(x, matrix_of_ones, iterations=erode_iterations)
        x = cv.dilate(x, matrix_of_ones, iterations=dilate_iterations)
    return x


def dilation_erosion(x, n, kernel_size=3, erode_iterations=1, dilate_iterations=1):
    matrix_of_ones = np.ones((kernel_size, kernel_size), np.uint8)
    for _ in range(n):
        x = cv.dilate(x, matrix_of_ones, iterations=dilate_iterations)
        x = cv.erode(x, matrix_of_ones, iterations=erode_iterations)
    return x


def processed(image):
    grayscaled = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(grayscaled, (5, 5), 0)
    thresholded = cv.Canny(blurred, threshold1=200, threshold2=200)
    return thresholded


# https://stackoverflow.com/questions/37177811/crop-rectangle-returned-by-minarearect-opencv-python
def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
               pts[1][0]:pts[2][0]]

    return img_crop


# https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python/48553593#48553593
def getSubImage(src, rect):
    # Get center, size, and angle from rect
    center, size, theta = rect
    # Convert to int
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D(center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, src.shape[:2])
    out = cv2.getRectSubPix(dst, size, center)
    return out


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
    return {
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
    imgray = cv.erode(imgray, np.ones((3,3), np.uint8), iterations=1)
    imgray = erosion_dilation(imgray, 10)
    imgray = cv.GaussianBlur(imgray, (5, 5), 0, borderType=cv.BORDER_CONSTANT)
    imgray = cv.GaussianBlur(imgray, (5, 5), 0, borderType=cv.BORDER_CONSTANT)

    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    thresh = erosion_dilation(thresh, 10)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(contours):
        img_copy = img.copy()
        # get the bounding rect
        # x, y, w, h = cv.boundingRect(c)
        rect = cv.minAreaRect(c)
        img_cropped = getSubImage(img_copy, rect)
        if(img_cropped.shape[0] < img_cropped.shape[1]):
            img_cropped = cv.rotate(img_cropped, cv.ROTATE_90_CLOCKWISE)
        # to save the images
        cv.imwrite('./cards_cropped_from_img/img_{}.jpg'.format(i), img_cropped)
        # cv.imwrite('./cards_cropped_from_img/img_{}.jpg'.format(i), img[y:y+h,x:x+w])

    # cv.putText(img, "jakis tekst", (500,375), cv.FONT_HERSHEY_COMPLEX, 1.0, WHITE, 2)

    print(len(contours))  # number of detected countours, ~ ideally number of detected cards
    cv.drawContours(img, contours, -1, GREEN, 3)

    # Displaying the results
    cv.imshow('', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # TODO: switch to matplotlib for showing multiple images
    return img


def main(path):
    images = OrderedDict({})
    for (root, _, filenames) in os.walk(path):
        for filename in filenames:
            images[filename] = card_contouring(root + filename)

    """ Ponizszy fragment laczy i wyswietla obrazki horyzontalnie"""
    # tmp = np.hstack((img, img))
    # cv.imshow("Obrazki na horyzoncie", tmp)
    for imagename in images.keys():
        show(images[imagename])

    # TODO: Dipslay multiple images https://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/


def process_selected_images(path, selected_images):
    for filename in selected_images:
        card_contouring(path + filename)


if __name__ == '__main__':
    WIDTH = 750
    HEIGHT = 750
    PATH = './karty/'

    selected_images = ['dwie_pary.jpg', 'dwie.jpg', 'as_pik.jpg', 'krol_karo.jpg']
    process_selected_images(PATH,
                            selected_images)  # FIXME: processes only the first element in the list, due to cv.waitKey(0) and cv.destroyAllWindows()
