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


def to_canny(src, threshold):
    return cv.Canny(src, threshold, threshold * 2)


def find_contours(canny_src, visualise=False):
    contours, hierarchy = cv.findContours(
        canny_src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    if visualise:
        drawing = np.zeros((canny_src.shape[0], canny_src.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
        # Show in a window
        cv.imshow("Contours", drawing)

    return contours

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
    # dest = to_blur(dest)
    show_image("Grayscale + Blur", dest)

    canny_output = to_canny(dest, threshold)
    show_image("Canny", canny_output)

    contours = find_contours(canny_output, visualise=True)

    cv.waitKey()


if __name__ == "__main__":
    main(sys.argv[1:])
