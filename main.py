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
    img = cv.imread("karty/dwie_pary.jpg")

    width = 1000
    height = 750
    dim = (width, height)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    thresh = erosion_dilation(thresh, 10)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.putText(img, "jakis tekst", (500,375), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
    print(len(contours)) # number of detected countours, ~ ideally number of detected cards
    cv.drawContours(img, contours, -1, GREEN ,3)


    cv.imshow("Karta", img)

    cv.waitKey(0)
    cv.destroyAllWindows()