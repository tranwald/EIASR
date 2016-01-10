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
print(*resized.shape, sep=', ')

def repeat_bilateral_filter(src, n, *args, **kwargs):
    filtered = src
    for _ in itertools.repeat(None, n):
        filtered = cv2.bilateralFilter(filtered, *args, **kwargs)
    return filtered

def on_change_bilateral(x):
    tb_pos = cv2.getTrackbarPos(tb_name_filter, window_name)
    filtered = repeat_bilateral_filter(resized, tb_pos, d=9, sigmaColor=9, sigmaSpace=7)
    print('Iterations for billateal filter', tb_pos)
    cv2.imshow("Transformed", filtered)

def on_change_median(x):
    tb_pos = cv2.getTrackbarPos(tb_name_filter, window_name)
    tb_pos = tb_pos if tb_pos % 2 == 1 else tb_pos + 1
    filtered = cv2.medianBlur(resized, ksize=tb_pos)
    cv2.imshow("Transformed", filtered)    

# show images
window_name = "Transformed"
tb_name_bilateral = "Bilateral Filter"
tb_name_median = "Median Filter"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.createTrackbar(tb_name_bilateral, window_name, 0, 100, on_change_median)
cv2.createTrackbar(tb_name_median, window_name, 0, 20, on_change_bilateral)
cv2.imshow("Transformed", resized)

# repeat_bilateral_filter(resized, tb_pos, d=9, sigmaColor=9, sigmaSpace=7)

# cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
# cv2.imshow("Original", orig)

cv2.waitKey(0)
cv2.destroyAllWindows()