import easyocr
import cv2
from collections import Counter


test = [1,1,1,2,2,3,3,3,3]

plate = Counter(test).most_common(1)[0][0] #gets majority vote
print(plate)