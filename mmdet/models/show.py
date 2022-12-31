import cv2
import numpy

img = cv2.imread('/disk0/evannnnnnn/home/Datasets/DOTA/train/images/P0896.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
lbl = open('/disk0/evannnnnnn/home/Datasets/DOTA/train/labelTxt/P0896.txt')
lbl.readline()
lbl.readline()

for l in lbl:
    x1, y1, x2, y2, x3, y3, x4, y4, c, _ = l.split()
    x1, y1, x2, y2, x3, y3, x4, y4 =\
        int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)
    b = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    if c == 'harbor': color = (255, 255, 0)
    if c == 'ship': color = (0, 255, 255)
    cv2.polylines(img, [numpy.array(b)], isClosed=True, color=color, thickness=2)

cv2.imwrite('/disk0/evannnnnnn/home/P0896.jpg', img)
cv2.imshow('img', img)
# cv2.waitKey()