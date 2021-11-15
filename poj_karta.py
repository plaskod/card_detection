import cv2
import numpy as np

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


if __name__ == '__main__':
    img = cv.imread("cards_cropped_from_img/img_1.jpg")

    width = 1000
    height = 750
    dim = (width, height)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 140, 255, 0)
    thresh = erosion_dilation(thresh, 5)
    thresh = cv.erode(thresh, np.ones((3, 3), np.uint8), iterations=3)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.putText(img, "jakis tekst", (500,375), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
    print(len(contours)) # number of detected countours, ~ ideally number of detected cards
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
        print(cX, cY)
        if cX < 145 and cY < 200:
            cv.drawContours(img, contours, i, GREEN ,3)


    cv.imshow("Karta", img)

    cv.waitKey(0)
    cv.destroyAllWindows()
