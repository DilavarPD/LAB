import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('otsu_1.jpg')
#image=cv2.blur(img, (5,5))
cv2.bilateralFilter(img)

i = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
cv2.imshow("f",i)
cv2.waitKey(0)
cv2.destroyAllWindows()




