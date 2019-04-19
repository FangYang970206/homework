import cv2
import argparse
import numpy as np


def hazeRemoval(img, w=0.7, t0=0.1):
    #求每个像素的暗通道
    darkChannel = img.min(axis=2)
    #取暗通道的最大值最为全球大气光
    A = darkChannel.max()
    darkChannel = darkChannel.astype(np.double)
	#利用公式求得透射率
    t = 1 - w * (darkChannel / A)
	#设定透射率的最小值
    t[t < t0] = t0

    J = img
    #对每个通道分别进行去雾
    J[:, :, 0] = (img[:, :, 0] - (1 - t) * A) / t
    J[:, :, 1] = (img[:, :, 1] - (1 - t) * A) / t
    J[:, :, 2] = (img[:, :, 2] - (1 - t) * A) / t
    return J

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    args = vars(ap.parse_args())

    hazeImage = cv2.imread(args["image"])

    result = hazeRemoval(hazeImage.copy())

    cv2.imshow("HazeRemoval", np.hstack([hazeImage, result]))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
