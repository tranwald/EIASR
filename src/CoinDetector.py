#!/usr/bin/env python3

import cv2
import argparse
import numpy as np
import itertools
import sys

def repeat_bilateral_filter(src, n, *args, **kwargs):
    filtered = src
    for _ in itertools.repeat(None, n):
        filtered = cv2.bilateralFilter(filtered, *args, **kwargs)
    return filtered

def imfill(src):
    floodfill = src.copy()
    height, width = floodfill.shape
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(floodfill, mask, (0, height - 2), 255)
    floodfill_inv = cv2.bitwise_not(floodfill)
    return src | floodfill_inv

def fixed_height_resize(src, new_height):
    ratio = height/new_height
    resize_ratio = ratio if ratio < 1 else 1/ratio
    resized = cv2.resize(src, (0,0), fx=resize_ratio, fy=resize_ratio)
    return resized

def min_max_dict(src):
    min_max_keys = ['minVal', 'maxVal', 'minLoc', 'maxLoc']
    min_max_values = list(cv2.minMaxLoc(src))
    return dict(zip(min_max_keys, min_max_values))

def auto_edge_detection(src, sigma=0.33):
    median = np.median(src)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(src, lower, upper)

# set up arg parser & read args
parser = argparse.ArgumentParser(description="EIASR 2015Z project")
parser.add_argument("-i", "--image", required=True, 
                    help="Path to an input image used in coin detection task")
args = parser.parse_args()
print("Image path: ", args.image)
# read image from path given as arg
image = cv2.imread(args.image)
if image is None:
    print("Failed to load image file:", args.image)
    sys.exit(1)
orig = image.copy()
# get image width and heigh
height, width, _ = image.shape
print("Image dimensions (w, h): ", end="")
print(height, width, sep=', ')
# convert to a grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# resize to speedup bilateral filter
resized = fixed_height_resize(gray, 480)
print("Resized image dimensions: ", end="")
print(*resized.shape, sep=', ')
# apply bilateral filter 10 times
filtered = repeat_bilateral_filter(resized, 5, d=9, sigmaColor=9, sigmaSpace=7)
# apply median filter to reduce pepper and salt noise
filtered = cv2.GaussianBlur(filtered, ksize=(3, 3), sigmaX=0)
# apply Otsu bizarization
ret, thresheld = cv2.threshold(filtered, 127, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# set up windows for showing results
window_trans = "Transformed"
window_prev = "Previous transformations"
cv2.namedWindow(window_trans, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_prev, cv2.WINDOW_NORMAL)
# tb_name_opening = "Opening"
# tb_name_iters = "Iterations"
# nothing = lambda *args: None
# cv2.createTrackbar(tb_name_opening, window_name, 0, 100, nothing)
# cv2.createTrackbar(tb_name_iters, window_name, 1, 100, nothing)
# set up kernel for morphological transformations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(4, 4))
# dilate to close gaps in conturs
morphed = cv2.dilate(thresheld, kernel, iterations=2)
# fill holes in coins by using Canny edge detector
canny = auto_edge_detection(morphed)
filled = imfill(canny) | morphed
# erode to reduce coins sizes
eroded = cv2.erode(filled, kernel, iterations=2)
# apply distance transform to create mask for watershed
dist_transform = cv2.distanceTransform(eroded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
dist_norm = cv2.normalize(dist_transform, 0.0, 100.0, cv2.NORM_MINMAX)
# show images
cv2.imshow(window_trans, np.hstack([thresheld, dist_norm]))
prev_list = [resized, filtered, thresheld, morphed, filled, eroded]
cv2.imshow(window_prev, np.hstack(prev_list))
# wait for ESC to close
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows() 