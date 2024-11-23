"""
@file task.py
@brief Count the objects in one single image. For every object get the following properties:
* Coordinates of the center of the object
* Aspect Ratio
* Extent
* Solidity
* Equivalent Diameter
* Orientation
* Mean Intensity
"""

import sys
import cv2 as cv
from os import path


def show_image(window_name, src):
    cv.namedWindow(window_name)
    cv.imshow(window_name, src)


def load_input_image(image):
    # Load the image
    src = cv.imread(image, cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print("Error opening image: " + image)
        exit(-1)
    # Show source image
    show_image("Source", src)
    return src


def main(argv):
    # [load_image]
    # Check number of arguments
    if len(argv) < 1:
        print("Not enough parameters")
        print("Usage:\task.py < path_to_image >")
        return -1
    # Load the image
    src = load_input_image(argv[0])
    cv.waitKey()

    # canny_output = cv.Canny(src_gray, threshold, threshold * 2)


if __name__ == "__main__":
    main(sys.argv[1:])
