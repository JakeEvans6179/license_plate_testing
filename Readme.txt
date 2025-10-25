Fine tuned yolov8n model trained on dataset of license plates

Once license plate is recognised OCR is applied to extract the number plate value

set confidence level for tracking to begin at 40% to minimise false positives

test video for script - https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/


pip install opencv-python
pip install ultralytics
pip install easyocr