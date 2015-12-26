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
# apply gaussian blur
gauss = cv2.GaussianBlur(gray, ksize=(15, 15), sigmaX=0)
# show images
cv2.namedWindow("Transformed", cv2.WINDOW_NORMAL)
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow("Transformed", gauss)
cv2.imshow("Original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()