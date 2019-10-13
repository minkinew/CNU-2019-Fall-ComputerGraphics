import cv2
import numpy as np

def my_bgr2gray(img):
    result = (img[:,:,0] * 0.114) + (img[:,:,1] * 0.587) + (img[:,:,2] * 0.299)
    result = result.astype(np.uint8)

    return result

src = cv2.imread('D:\\py_data\\lena.png', cv2.IMREAD_COLOR)

gray = my_bgr2gray(src)

cv2.imshow('gray', gray)
cv2.waitKey()
cv2.destroyAllWindows()
