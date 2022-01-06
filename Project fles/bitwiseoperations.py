#import opencv
import cv2
import cv2 as cv
#read the images
img1 = cv.imread('cannypik/1.jpg')
img2 = cv.imread('otsu_1.jpg')
bitwise_AND = cv.bitwise_and(img1, img2)
bitwise_OR = cv.bitwise_or(img1, img2)
bitwise_NOT = cv.bitwise_not(img2)


s1 = cv.resize(img1, (0, 0), fx=0.3, fy=0.3)
s2 = cv.resize(img2, (0, 0), fx=0.3, fy=0.3)
s3 = cv.resize(bitwise_AND, (0, 0), fx=0.3, fy=0.3)
s4 = cv.resize(bitwise_OR, (0, 0), fx=0.3, fy=0.3)
s5 = cv.resize(bitwise_NOT, (0, 0), fx=0.3, fy=0.3)
cv.imshow('img1',s1)
cv.imshow('img2',s2)
cv.imshow('AND',s3)
cv.imshow('OR',s4)
cv.imshow('NOT',s5)
cv.imwrite("AND_1.jpg",s3)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()