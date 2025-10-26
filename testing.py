import cv2
import easyocr
img = cv2.imread('image.jpg', 0)
blur = cv2.GaussianBlur(img,(5,5),0)
reader = easyocr.Reader(['en'])
result = reader.readtext(blur)
for (bbox, text, prob) in result:
    print(f'Text: {text}, Probability: {prob}')