import numpy as np
import cv2
from skimage import util

img = cv2.imread('otsu_11.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)  # Use cv2.CCOMP for two level hierarchy
#cv2.drawContours(img, contours, -1, (0,255,0), 2)

# create an empty mask
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# loop through the contours
for i, cnt in enumerate(contours):
    # if the contour has no other contours inside of it

    if hierarchy[0][i][3] != -1:  # basically look for holes
        # if the size of the contour is less than a threshold (noise)
        if cv2.contourArea(cnt) < 70:
            # Fill the holes in the original image
            cv2.drawContours(img, [cnt], 0, (0,0,0), -1)


i1 = cv2.resize(mask, (0, 0), fx=0.3, fy=0.3)
i2 = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)

cv2.imshow("Mask", i1)
cv2.imshow("Img", i2)
image = cv2.bitwise_not(img, img, mask=mask)
i3 = cv2.resize(mask, (0, 0), fx=0.3, fy=0.3)
i4 = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
cv2.imshow("Mask", i3)
cv2.imshow("After", i4)

cv2.waitKey()
cv2.destroyAllWindows()