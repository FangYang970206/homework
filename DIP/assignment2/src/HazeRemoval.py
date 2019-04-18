import cv2
import argparse
import numpy as np
from PIL import Image

def hazeRemoval(img, windowSize=15, w=0.95, t0=0.1):
    darkChannel = img.min(axis=2)
    A = darkChannel.max()
    darkChannel = darkChannel.astype(np.double)

    t = 1 - w * (darkChannel / A)

    t[t < t0] = t0

    J = img
    J[:, :, 0] = (img[:, :, 0] - (1 - t) * A) / t
    J[:, :, 1] = (img[:, :, 1] - (1 - t) * A) / t
    J[:, :, 2] = (img[:, :, 2] - (1 - t) * A) / t
    return J

def haze_removal(image, windowSize=24, w0=0.6, t0=0.1):

    darkImage = image.min(axis=2)
    maxDarkChannel = darkImage.max()
    darkImage = darkImage.astype(np.double)

    t = 1 - w0 * (darkImage / maxDarkChannel)
    T = t * 255
    T.dtype = 'uint8'

    t[t < t0] = t0

    J = image
    J[:, :, 0] = (image[:, :, 0] - (1 - t) * maxDarkChannel) / t
    J[:, :, 1] = (image[:, :, 1] - (1 - t) * maxDarkChannel) / t
    J[:, :, 2] = (image[:, :, 2] - (1 - t) * maxDarkChannel) / t
    result = Image.fromarray(J)

    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    args = vars(ap.parse_args())

    # hazeImage = cv2.imread(args["image"])
    hazeImage = np.array(Image.open(args["image"]))
    # result1 = hazeRemoval(hazeImage)
    result2 = haze_removal(hazeImage)

    # cv2.imshow("HazeRemoval", np.hstack([hazeImage, result1]))
    cv2.imshow("HazeRemoval", np.hstack([hazeImage, result2]))
    cv2.waitKey(0)


# if __name__ == '__main__':
#     main()
if __name__ == '__main__':
    imageName = "20.jpg"
    image = np.array(Image.open(imageName))
    imageSize = image.shape
    result = haze_removal(image)
    cv2.imshow("HazeRemoval", np.hstack([image, result]))
    cv2.waitKey(0)
    # result.show()