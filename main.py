import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    from cv2 import cv2 as cv
except ImportError:
    pass




BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)



""" PRZETWARZANIE OBRAZU """
class Board():
    def __init__(self, filepath_):
        self.filepath = filepath_
        self.width = WIDTH
        self.height = HEIGHT
        self.img = self.read_img(self.filepath)
        self.detected_contours, self.detected_cards = self.detect_cards_on_board()

    def read_img(self, filepath):
        img = cv.imread(filepath)
        global WIDTH, HEIGHT
        dim = (WIDTH, HEIGHT)
        img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        return img
    
    def detect_cards_on_board(self):
        img = self.img
        imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgray = cv.erode(imgray, np.ones((3,3), np.uint8), iterations=1)
        imgray = erosion_dilation(imgray, 10)
        imgray = cv.GaussianBlur(imgray, (5, 5), 0, borderType=cv.BORDER_CONSTANT)
        imgray = cv.GaussianBlur(imgray, (5, 5), 0, borderType=cv.BORDER_CONSTANT)

        ret, thresh = cv.threshold(imgray, 140, 255, 0)
        thresh = erosion_dilation(thresh, 10)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cards = []
        for c in contours:
            img_cropped, _ = crop_image(img, c)
            card = Card(c, img_cropped)
            card.nadajFigure()
            card.nadajKolor()
            cards.append(card)
        # Displaying the results
        # cv.drawContours(img, contours, -1, GREEN, 3)
        # cv.imshow('', img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    
        return contours, cards

    def label_cards(self):
        img = self.img
        assert self.detected_contours is not None
        for i,c  in enumerate(self.detected_contours):
            cv.drawContours(img, self.detected_contours, i, MAGENTA, 3)
            x, y, _, _ = cv.boundingRect(c)
            label = "{} {}".format(self.detected_cards[i].figura, self.detected_cards[i].kolor)
            translated_label = ' '.join([card_label_names.get(i, i) for i in label.split()]) # tlumaczenie znakow A, K, Q na As, Krol, Dama
            cv.putText(img, translated_label, (x, y-10), cv.FONT_HERSHEY_COMPLEX, 0.8, WHITE, 2)
        show(img)


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

card_label_names = {
    'A' : 'As',
    'K' : 'Krol',
    'Q' : 'Dama',
    'J' : 'Walet',
}

def wczytaj(img):
    thresh = cv.imread(img)
    thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(thresh, 140, 255, 0)
    return thresh

maski_figur = {
    "9": wczytaj("template/9.jpg"),
    "10": wczytaj("template/10.jpg"),
    "J": wczytaj("template/J.jpg"),
    "Q": wczytaj("template/Q.jpg"),
    "K": wczytaj("template/K.jpg"),
    "A": wczytaj("template/A.jpg")
}

maski_koloru = {
    "kier": wczytaj("template/kier.jpg"),
    "karo": wczytaj("template/karo.jpg"),
    "pik": wczytaj("template/pik.jpg"),
    "trefl": wczytaj("template/trefl.jpg"),
}

class Card:
    def __init__(self, contour_, img_):
        self.contour = contour_
        self.img = img_
        self.figura = ''
        self.kolor = ''
        self.pozycja = None

    def nadajFigure(self):
        width = 500
        height = 500
        dim = (width, height)
        img = cv.resize(self.img, dim, interpolation=cv.INTER_AREA)
        imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 140, 255, 0)
        thresh = erosion_dilation(thresh, 5)
        thresh = cv.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.putText(img, "jakis tekst", (500,375), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
        # number of detected countours, ~ ideally number of detected cards
        flaga = 0
        for i in range(len(contours)):
            M = cv2.moments(contours[i])
            try:
                cX = int(M["m10"] / M["m00"])
            except ZeroDivisionError:
                cX = int(M["m10"] / 1)

            try:
                cY = int(M["m01"] / M["m00"])
            except ZeroDivisionError:
                cY = int(M["m01"] / 1)
            if cX < 100 and cY < 100 and cX != 0 and cY != 0 and flaga == 0 and hierarchy[0][i][3] == 0:
                rect = cv.boundingRect(contours[i])
                koord = [rect[0], rect[1]]
                size = [rect[2], rect[3]]
                wyciecie = cv.getRectSubPix(img, size, (koord[0]+size[0]//2, koord[1]+size[1]//2))
                cv.drawContours(img, contours, i, MAGENTA, 3)
                flaga = 1

        wyciecie = cv.resize(wyciecie, (60, 80), interpolation=cv.INTER_AREA)
        wyciecie = cv.cvtColor(wyciecie, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(wyciecie, 140, 255, 0)
        thresh = cv.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

        najlepszy = 10000
        naj_figura = "9"
        for figura, maska in maski_figur.items():
            diff_img = cv.absdiff(thresh, maska)
            rank_diff = int(np.sum(diff_img)/255)
            if rank_diff < najlepszy:
                najlepszy = rank_diff
                naj_figura = figura

        self.figura = naj_figura

    def nadajKolor(self):
        width = 500
        height = 500
        dim = (width, height)
        img = cv.resize(self.img, dim, interpolation=cv.INTER_AREA)
        imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 140, 255, 0)
        thresh = erosion_dilation(thresh, 5)
        thresh = cv.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.putText(img, "jakis tekst", (500,375), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
        # number of detected countours, ~ ideally number of detected cards
        flaga = 0
        for i in range(len(contours)):
            M = cv2.moments(contours[i])
            try:
                cX = int(M["m10"] / M["m00"])
            except ZeroDivisionError:
                cX = int(M["m10"] / 1)

            try:
                cY = int(M["m01"] / M["m00"])
            except ZeroDivisionError:
                cY = int(M["m01"] / 1)
            if cX < 100 and 90 < cY < 150 and cX != 0 and cY != 0 and hierarchy[0][i][3] == 0:
                rect = cv.boundingRect(contours[i])
                koord = [rect[0], rect[1]]
                size = [rect[2], rect[3]]
                wyciecie = cv.getRectSubPix(img, size, (koord[0] + size[0] // 2, koord[1] + size[1] // 2))
                cv.drawContours(img, contours, i, MAGENTA, 3)
                flaga = 1

        wyciecie = cv.resize(wyciecie, (60, 80), interpolation=cv.INTER_AREA)
        wyciecie = cv.cvtColor(wyciecie, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(wyciecie, 140, 255, 0)
        thresh = cv.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

        najlepszy = 10000
        naj_kolor = "kier"
        for kolor, maska in maski_koloru.items():
            diff_img = cv.absdiff(thresh, maska)
            rank_diff = int(np.sum(diff_img) / 255)
            if rank_diff < najlepszy:
                najlepszy = rank_diff
                naj_kolor = kolor

        self.kolor = naj_kolor

    def getFigura(self):
        return self.figura
    
    def getKolor(self):
        return self.kolor
    
    def getPozycja(self):
        return self.pozycja

def show(image):
    cv.imshow("Karta", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def crop_image(img, contour):
    rect = cv.minAreaRect(contour)
    img_cropped = getSubImage(img, rect)
    if(img_cropped.shape[0] < img_cropped.shape[1]):
        img_cropped = cv.rotate(img_cropped, cv.ROTATE_90_CLOCKWISE)

    return img_cropped, rect

WIDTH = 750
HEIGHT = 750
PATH = './karty_m/ustawienia/'
def main():
    selected_images = ['flush.jpg', 'four_of_kind.jpg', 'full.jpg', 'hight_card.jpg', 'pair.jpg', 'pair2.jpg', 'royal_flush.jpg', 'small_strit.jpg',
        'three_of_kind.jpg', 'two_pairs.jpg', 'two_pairs2.jpg']

    for filename in [selected_images[4]]:
        board = Board(PATH+filename) # detekcja kart w konstruktorze
        board.label_cards() # tutaj jest odkomentowany show(img)
        print('----------------------------------')

if __name__ == '__main__':
    main()
