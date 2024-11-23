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
    return src


def to_grayscale(src):
    return cv.cvtColor(src, cv.COLOR_BGR2GRAY)


def to_blur(src, kernel_size=(3, 3)):
    return cv.blur(src, kernel_size)


def find_contours(src, threshold):
    canny_output = cv.Canny(src, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    return contours, canny_output

def main(argv):
    # [load_image]
    # Check number of arguments
    if len(argv) < 1:
        print("Not enough parameters")
        print("Usage:\task.py <path_to_image> [<threshold>]")
        return -1
    # Load the image
    src = load_input_image(argv[0])
    show_image("Source", src)

    # Set the threshold
    threshold = 70 if len(argv) < 2 else int(argv[1])

    # Convert image to gray and blur it
    dest = to_grayscale(src)
    dest = to_blur(dest)
    show_image("Grayscale + Blur", dest)

    contours, dest = find_contours(dest, threshold)
    show_image("Canny", dest)

    cv.waitKey()


if __name__ == "__main__":
    main(sys.argv[1:])
