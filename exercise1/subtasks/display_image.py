import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("data/profile-pic.jpg"))

if img is None:
    sys.exit("Could not read the provided image!")

cv.imshow("Display window", img)
keyPerssed = cv.waitKey(0)

# If "s" button on keyboard is pressed
if keyPerssed == ord("s"):
    cv.imwrite("dist/output_img.png", img=img)
