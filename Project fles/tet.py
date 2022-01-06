import cv2

image = cv2.imread("cannypik/1.jpg")
img = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
edges = cv2.Canny(img, 100, 200)
cv2.imshow("Edge Detected Image", edges)
cv2.imshow("Original Image", img)

cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image