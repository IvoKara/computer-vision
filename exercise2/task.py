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

import os
import sys
import json
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
        'Press "s" key to save the results in "dist" folder.',
    )
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
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

    return contours, drawing


def show_object_props(object_props):
    for index, props in enumerate(object_props, start=1):
        print(f"Object {index}:")
        for prop_name, prop_value in props.items():
            print(f"  {prop_name}: {prop_value}")
        print()


def get_object_props(contours, src):
    for contour in contours:
        # Compute center of the object (centroid)
        M = cv.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = None, None

        # Compute bounding rectangle
        x, y, w, h = cv.boundingRect(contour)

        # Compute aspect ratio
        aspect_ratio = w / h

        # Compute area and perimeter
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)

        if area == 0:  # Avoid division by zero
            continue

        # Compute extent (area / bounding rectangle area)
        rect_area = w * h
        extent = area / rect_area

        # Compute convex hull and solidity (area / convex hull area)
        hull = cv.convexHull(contour)
        hull_area = cv.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Compute equivalent diameter
        equivalent_diameter = np.sqrt(4 * area / np.pi)

        # Fit ellipse and compute orientation if applicable
        if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
            ellipse = cv.fitEllipse(contour)
            orientation = ellipse[2]
        else:
            orientation = None

        # Compute mean intensity of the object
        mask = np.zeros(src.shape, dtype=np.uint8)
        cv.drawContours(mask, [contour], -1, 255, thickness=-1)
        mean_intensity = cv.mean(src, mask=mask)[0]

        yield {
            "center": [cx, cy],
            "aspect_ratio": aspect_ratio,
            "extent": extent,
            "solidity": solidity,
            "equivalent_diameter": equivalent_diameter,
            "orientation": orientation,
            "mean_intensity": mean_intensity,
        }


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

    contours, drawings = find_contours(canny_output, visualise=True)

    object_props = get_object_props(contours, dest)
    show_object_props(object_props)

    show_output_message()

    # if "s" key is pressed
    # Save the contour image
    # and the object props as json file
    if cv.waitKey(0) == ord("s"):
        directory = "dist"
        if not os.path.exists(directory):
            os.makedirs(directory)

        cv.imwrite(os.path.join(directory, "coutours.png"), drawings)

        with open(os.path.join(directory, "object_props.json"), "w") as json_file:
            json.dump(list(object_props), json_file)


if __name__ == "__main__":
    main(sys.argv[1:])
