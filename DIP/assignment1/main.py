import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('--image', required=True)
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

eh_image = cv2.equalizeHist(image)

ahe = cv2.createCLAHE(clipLimit=0.0, tileGridSize=(8,8))
ahe_image = ahe.apply(image)

clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
clahe_image = clahe.apply(image)

cv2.imshow("Histogram Equalization", np.hstack([image, eh_image, 
                                                ahe_image, clahe_image]))
cv2.waitKey(0)