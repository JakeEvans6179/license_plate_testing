from ultralytics import YOLO
import cv2
import easyocr
import numpy as np

conf_threshold = 0.4 #only include confidence level >= 40%


#custom yolo model
model = YOLO('license_plate_best.pt')

#test video
video_path = 'traffic_cam.mp4'

#open video
capture = cv2.VideoCapture(video_path)

if not capture.isOpened():
    print(f"Error: Cannot open video file at {video_path}")
    exit()

#video properties
fps = capture.get(cv2.CAP_PROP_FPS)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video Info: {width}x{height} at {fps:.2f} FPS")

#read and process frames
while capture.isOpened():
    ret, frame = capture.read()

    if not ret:
        print("End of video or failed to read frame")
        break



    #apply model tracking
    results = model.track(frame, persist=True) #applies model to track frame and also prints out no. plates tracked in the frame

    detections = results[0]

    for detection in detections.boxes.data.tolist():
        print(detection) #prints x1, y1, x2, y2, trackid, conflvl, classid






    boxes = results[0].boxes

    high_conf_mask = boxes.conf >= conf_threshold #create a mask for boxes with confidence level higher than our defined value

    filtered_boxes = boxes[high_conf_mask] #applied mask to create new box object with only filtered results

    results[0].boxes = filtered_boxes



    #plot results on screen
    frame_plot = results[0].plot()

    #show the processed frame
    cv2.imshow('License Plate Detection', frame_plot)


    #exit when q pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#end script
capture.release()
cv2.destroyAllWindows()