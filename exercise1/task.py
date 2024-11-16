import cv2 as cv
from os import path

# Global Defaults
KERNEL_SIZE = 3
IMG_SOURCE = "data/forest.jpg"
OUTPUT_IMAGE = "dist/output.png"
DDEPTH = cv.CV_16S
THRESHOLD_VALUE = 50
# Thresold type 1: Binary Inverted
THRESHOLD_TYPE = 1
MAX_BINARY_VALUE = 255

source = None
destination = None


def main():
    cwd = path.dirname(path.realpath(__file__))
    img_path = path.join(cwd, IMG_SOURCE)
    source = cv.imread(cv.samples.findFile(img_path))
    if source is None:
        print("Error opening image")
        return -1

    cv.namedWindow("Original", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("After Median Blur", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("After Grayscale", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("After Laplace", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Result Image", cv.WINDOW_AUTOSIZE)

    global destination
    cv.imshow("Original", source)
    # Apply Median Blur
    destination = cv.medianBlur(source, KERNEL_SIZE)
    cv.imshow("After Median Blur", destination)

    # Convert to Grayscale
    destination = cv.cvtColor(destination, cv.COLOR_BGR2GRAY)
    cv.imshow("After Grayscale", destination)

    destination = cv.Laplacian(destination, DDEPTH, ksize=KERNEL_SIZE)
    destination = cv.convertScaleAbs(destination)
    cv.imshow("After Laplace", destination)

    _, destination = cv.threshold(
        destination, THRESHOLD_VALUE, MAX_BINARY_VALUE, THRESHOLD_TYPE
    )
    cv.imshow("Result Image", destination)

    # Wait to press any key
    cv.waitKey(0)

    out_path = path.join(cwd, OUTPUT_IMAGE)
    cv.imwrite(out_path, destination)


if __name__ == "__main__":
    main()
