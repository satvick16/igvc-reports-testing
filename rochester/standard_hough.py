import numpy as np
import cv2 as cv
import math
import sys

# Hough transform (detects straight lines)
# probabilistic Hough transform
# https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html

# edge-detection pre-processing is desirable before using Hough
# uses polar system


def standard(dst, cdst, t):
    # standard hough line transform arguments
    # dst: output of edge detector
    # rho: resolution of r (1 pixel used here)
    # theta: resolution of theta (1 degree)
    # TODO this param can be messed with : threshold: minimum # of intersections to "detect" a line (150)
    # TODO what even are these : srn, stn
    # lines: vector that will store params of detected lines

    # 150-200 neighbourhood seems to be optimal
    # the model sucks at ignoring extra info

    lines = cv.HoughLines(dst, 1, np.pi / 180, t, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

            cv.line(cdst, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)

    # saves output
    cv.imwrite(
        f"C:\\Users\\satvi\\Desktop\\igvc-tests\\rochester\\standard-model-output\\{t}threshold.jpg", cdst)


def main(argv):
    # load image file
    default_file = "C:\\Users\\satvi\\Desktop\\igvc-tests\\rochester\\test-img2.jpg"
    filename = argv[0] if len(argv) > 0 else default_file
    # convert to grayscale
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print(
            'Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    for t in range(100, 201, 5):
        # TODO play with this (try sobel) : canny detector does edge detection
        dst = cv.Canny(src, 50, 200, None, 3)
        # Copy edges to the images that will display the results in BGR
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        standard(dst, cdst, t)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
