from ultralytics import YOLO
import cv2
import easyocr
import numpy as np



conf_threshold = 0.4 #only include confidence level >= 40%

image_dictionary = {} #used to store all the cropped images used for OCR detection
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
        print(detection) #prints [x1, y1, x2, y2, trackid, conflvl, classid]






    boxes = results[0].boxes

    high_conf_mask = boxes.conf >= conf_threshold #create a mask for boxes with confidence level higher than our defined value

    filtered_boxes = boxes[high_conf_mask] #applied mask to create new box object with only filtered results

    results[0].boxes = filtered_boxes

    #cropping and processing for OCR
    if filtered_boxes.id is not None:
        for i in range(len(filtered_boxes)):

            box_coords = filtered_boxes[i].xyxy.cpu().numpy().astype(int) #extract coordinates of bounding box, moves tensor data from gpu to cpu and converts to integer data type
            track_id = filtered_boxes[i].id.cpu().numpy().astype(int)[0] #extract unique tracking id

            print("box_coords:",box_coords)
            print("track_id:",track_id)

            x1, y1, x2, y2 = box_coords[0] #unpack coordinates

            license_plate_crop = frame[y1:y2, x1:x2] #crop license plate from original frame

            if license_plate_crop.size == 0: #prevent empty crops
                continue


            #process cropped image for OCR
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV) #pixels below 64 go to 255, pixels above 64 go to 0 for OCR

            if track_id not in image_dictionary:
                image_dictionary[track_id] = [] #create list to hold the images with track id as the key

            image_dictionary[track_id].append(license_plate_crop_thresh)

            cv2.imshow(f"Processed Plate ID {track_id}", license_plate_crop_thresh)
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

print("Summary of Collected Images")
for track_id, images in image_dictionary.items():
    print(f"Car ID {track_id}: Collected {len(images)} processed images.")