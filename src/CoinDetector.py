import cv2
import argparse
import numpy as np

# set up arg parser & read args
parser = argparse.ArgumentParser(description="EIASR 2015Z project")
parser.add_argument("-i", "--image", required=True, 
                    help="Path to an input image used in coin detection task")
args = parser.parse_args()
print("Image path: ", args.image)
# read image from path given as arg
image = cv2.imread(args.image)
orig = image.copy()

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# equalize the histogram to remove light areas
equal = cv2.equalizeHist(gray)
# apply gaussian blur
gauss = cv2.GaussianBlur(equal, ksize=(7, 7), sigmaX=0)
# apply Otsu bizarization
ret, thresh = cv2.threshold(gauss, 127, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# perform series of erosions and diltions
# thresh = cv2.dilate(thresh, None, iterations=7)
# thresh = cv2.erode(thresh, None, iterations=7)
# show images
cv2.namedWindow("Transformed", cv2.WINDOW_NORMAL)
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow("Transformed", equal)
cv2.imshow("Original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()