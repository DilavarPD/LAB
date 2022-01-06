
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Import image
img = cv2.imread("cannypik/11.jpg")
se = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
cv2.imshow("org",se)
# Create float
bgr = img.astype(float)/255.

# Extract channels
with np.errstate(invalid='ignore', divide='ignore'):
	K = 1 - np.max(bgr, axis=2)
	C = (1-bgr[...,2] - K)/(1-K)
	M = (1-bgr[...,1] - K)/(1-K)
	Y = (1-bgr[...,0] - K)/(1-K)

# Convert the input BGR image to CMYK colorspace
CMYK = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)

# Split CMYK channels
Y, M, C, K = cv2.split(CMYK)

np.isfinite(C).all()
np.isfinite(M).all()
np.isfinite(K).all()
np.isfinite(Y).all()
s = cv2.resize(M, (0, 0), fx=0.3, fy=0.3)
cv2.imshow("m",s)
ret, thresh1 = cv2.threshold(M, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# the window showing output image
# with the corresponding thresholding
# techniques applied to the input image
o = cv2.resize(thresh1, (0, 0), fx=0.3, fy=0.3)
cv2.imshow('Otsu Threshold',o)

cv2.imwrite("otsu_11.jpg",thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()

