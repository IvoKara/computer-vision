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

import json
import sys
import cv2 as cv
import numpy as np
import random as rng


def show_image(window_name, src):
    cv.namedWindow(window_name)
    cv.imshow(window_name, src)


def show_output_message():
    width, height = 800, 600  # Set the size of the window
    text = (
        "Open the terminal to see",
        "the found object properties.",
        'See results in "dist" folder',
        "Press Enter key to finish.",
    )
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)

    x = 20
    y = 50
    background = np.zeros((height, width, 3), dtype=np.uint8)
    for line in text:
        cv.putText(
            background, line, (x, y), font, font_scale, text_color, font_thickness
        )
        y += 50

    show_image("Output", background)


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


def show_object_props(object_props):
    for index, props in enumerate(object_props, start=1):
        print(f"Object {index}:")
        for prop_name, prop_value in props.items():
            print(f"  {prop_name}: {prop_value}")
        print()


def get_object_props(contours):
    for contour in contours:
        # Compute bounding rectangle
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = w / h

        yield {"aspect_ratio": aspect_ratio}


def main(argv):
    # [load_image]
    # Check number of arguments
    if len(argv) < 1:
        print("Not enough parameters")
        print("Usage:\ntask.py <path_to_image> [<threshold>]")
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

    object_props = get_object_props(contours)
    show_object_props(object_props)

    show_output_message()

    # if "Enter" key is pressed
    # Save the contour image
    # and the object props as json file
    if cv.waitKey(0) == ord("\n"):
        cv.imwrite("dist/coutours.png", contours)

        with open("dist/object_props.json", "w"):
            json.dump(list(object_props))


if __name__ == "__main__":
    main(sys.argv[1:])
