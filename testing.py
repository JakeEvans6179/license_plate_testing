import cv2
import easyocr

# 1. Read the image from the file into a variable
img = cv2.imread('image.jpg')

# Check if the image was loaded successfully
if img is None:
    print("Error: Could not read image file. Check the path.")
else:
    # 2. Convert the image to grayscale
    license_plate_crop_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Apply a threshold to create a clean black-and-white image. This is perfect for OCR.
    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

    # 4. Initialize the OCR reader
    reader = easyocr.Reader(['en'])

    # 5. Feed the CLEAN, THRESHOLDED image to the reader (DO NOT BLUR)
    result = reader.readtext(license_plate_crop_thresh)

    # 6. Print the results
    print("Full result =", result)

    if not result:
        print("No text found.")
    else:
        for (bbox, text, prob) in result:
            print(f'Text: {text}, Probability: {prob:.4f}')