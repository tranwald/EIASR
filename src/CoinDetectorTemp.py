import cv2
import argparse
import numpy as np
import itertools

def repeat(f, N):
    for _ in itertools.repeat(None, N): f()

# set up arg parser & read args
parser = argparse.ArgumentParser(description="EIASR 2015Z project")
parser.add_argument("-i", "--image", required=True, 
                    help="Path to an input image used in coin detection task")
args = parser.parse_args()
print("Image path: ", args.image)
# read image from path given as arg
image = cv2.imread(args.image)
orig = image.copy()
# get image width and heigh
height, width, _ = image.shape
target = 480
ratio = height/target
resize_ratio = ratio if ratio < 1 else 1/ratio

print(*image.shape, sep=', ')
# convert to a grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# resize to speedup bilateral filter
resized = cv2.resize(gray, (0,0), fx=resize_ratio, fy=resize_ratio)
print(*resizde.shape, sep=', ')
# apply gaussian blur
# median = cv2.medianBlur(gray, ksize=15)
# gauss = cv2.GaussianBlur(gray, ksize=(25, 25), sigmaX=0)
# bilateral = median
# N = 10
# for _ in itertools.repeat(None, N):
#     bilateral = cv2.bilateralFilter(bilateral, d=9, sigmaColor=9, sigmaSpace=7)

# apply Otsu bizarization
# ret, thresh = cv2.threshold(bilateral, 0, 255,
#                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# show images
window_name = "Transformed"
tb_name_filter = "Filter"
nothing = lambda *args: None
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.createTrackbar(tb_name_filter, window_name, 1, 255, nothing)
median = gray
# Loop for get trackbar pos and process it
while True:
    # Get position in trackbar
    tb_pos = cv2.getTrackbarPos(tb_name_filter, window_name)
    # Apply blur
    tb_pos = tb_pos if tb_pos % 2 == 1 else tb_pos + 1
    # filtered = cv2.GaussianBlur(gray, ksize=(tb_pos, tb_pos), sigmaX=0)
    filtered = cv2.medianBlur(gray, ksize=tb_pos)
    # apply Otsu bizarization
    ret, thresh = cv2.threshold(filtered, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(11, 11))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 10);
    dilated = cv2.dilate(opened, None, iterations = 10)
    # closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, 10);
    # Show in window
    cv2.imshow("Transformed", thresh)

    # If you press "ESC", it will return value
    ch = cv2.waitKey(5)
    if ch == 27:
        break

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)

cv2.imshow("Original", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()