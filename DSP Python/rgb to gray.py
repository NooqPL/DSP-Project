import cv2
import os
from PIL import Image


img = cv2.imread("C:/Users/krzys/Downloads/sth.png")
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', gray_image)
cv2.waitKey(0)
cv2.imwrite('K:/LEGOs/Bricks/sth_gray.png', gray_image)




"""
path = "K:/LEGOs/Bricks/Orange_Brick"
dstpath = 'K:/LEGOs/Bricks/BRICKS_GRAY/Orange_gray'

files = os.listdir(path)

i = 0

for image in files:
    i += 0
    img = cv2.imread(os.path.join(path,image))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(dstpath,image),gray)

"""


"""
for i in range(5):
    img = cv2.imread(f'K:/LEGOs/Bricks/Black_Brick/black_{i}.png', (i*100))
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite(f'K:/LEGOs/Bricks/BRICKS_GRAY/Black_gray/black_gray_{i}.png',i*100, img = gray_image)
"""
