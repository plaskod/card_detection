import cv2
import numpy as np

try:
    from cv2 import cv2 as cv
except ImportError:
    pass

img = cv.imread("karty/as_trefl.jpg")

width = 1000
height = 750
dim = (width, height)
img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
for i in range(10):
    thresh = cv.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
    thresh = cv.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cv.putText(img, "jakis tekst", (500,375), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
print(len(contours))
cv.drawContours(img, contours, -1, (0,255,0),3)

cv.imshow("Karta", img)

cv.waitKey(0)
cv.destroyAllWindows()
