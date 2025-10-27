from ultralytics import YOLO
import cv2
import easyocr
from collections import defaultdict, deque, Counter

dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'B': '3',
                    'T': '7'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '3': 'B',
                    '7': 'T'}

conf_threshold = 0.4 #only include confidence level >= 40%

#creates rolling dictionary holding 15 of the latest frames, if key is called that isn't on the dictionary it calls lambda function
image_dictionary = defaultdict(lambda: deque(maxlen=25)) #used to store all the cropped images used for OCR detection

last_seen_frame = defaultdict(int) #stores frame in which track_id was last seen

majority_vote = [] #for final processed number plate

frame_count = 0 #counts the frames to be used for ocr

#load YOLO model
model = YOLO('license_plate_best.pt')

#test video
video_path = 'traffic_cam.mp4'

#open video
capture = cv2.VideoCapture(video_path)
#capture = cv2.VideoCapture(0) #webcam
allowlist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" #characters OCR can detect
reader = easyocr.Reader(['en'])#initialise OCR reader, english language


def check_plates(plate_text):

    text_collection = list(plate_text) #puts into a list format
    if len(text_collection) == 7:
        for i in [0,1]:
            #Should be alphabet here
            if text_collection[i].isdigit():    #if its a number
                if text_collection[i] in dict_int_to_char:
                    text_collection[i] = dict_int_to_char[text_collection[i]] #replace with the most likely character

        for i in [2,3]:
            #should be number here
            if text_collection[i].isalpha():
                if text_collection[i] in dict_char_to_int:
                    text_collection[i] = dict_char_to_int[text_collection[i]] #replace with the most likely character

        for i in [4,5,6]:
            #Should be alphabet here
            if text_collection[i].isdigit():    #if its a number
                if text_collection[i] in dict_int_to_char:
                    text_collection[i] = dict_int_to_char[text_collection[i]] #replace with the most likely character

        return "".join(text_collection) #join back together

    else:
        return None

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
            """
            if track_id not in image_dictionary:
                image_dictionary[track_id] = [] #create list to hold the images with track id as the key

                OCR_readings[track_id] = [] #add the OCR text from the plates

            image_dictionary[track_id].append(license_plate_crop_thresh) #append the image details onto the list

            result = reader.readtext(license_plate_crop_thresh) #read the text from processed image
            for (bbox, text, prob) in result:
                OCR_readings[track_id].append(text)  # append the image details onto the list
            """

            image_dictionary[track_id].append(license_plate_crop_thresh)

            last_seen_frame[track_id] = frame_count



            cv2.imshow(f"Processed Plate ID {track_id}", license_plate_crop_thresh)
    #plot results on screen
    frame_plot = results[0].plot()

    #show the processed frame
    cv2.imshow('License Plate Detection', frame_plot)

    frame_count += 1 #keep track of the frame number

    for track_id in list(last_seen_frame.keys()):   #creates static list to iterate through so doesn't crash
        frame_value = last_seen_frame[track_id]
        if frame_count - frame_value >= 50 and len(image_dictionary[track_id]) >= 10:
            #perform OCR on all the values in the image dictrionary for that specific trackid

            OCR_readings = [] #used to store temporary OCR readings used for majority voting

            for image in image_dictionary[track_id]:
                result = reader.readtext(image, allowlist=allowlist) #read image, only allowing alphabet and numbers to be read

                for (bbox, text, prob) in result:

                    corrected_plate = check_plates(text)
                    if corrected_plate is not None:
                        OCR_readings.append(corrected_plate) #append the value read from image, only append if correct length (ie not none) or else if majority becomes none script can crash

            del image_dictionary[track_id]  #remove the images from dictionary
            del last_seen_frame[track_id]   #remove last seen frame from track_id

            if OCR_readings:  #Only perform majority check if there are valid plates otherwise script would crash
                plate = Counter(OCR_readings).most_common(1)[0][0] #gets majority vote
                majority_vote.append(plate)





    #exit when q pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#end script
capture.release()
cv2.destroyAllWindows()

print("Summary of Collected Images")
for track_id, images in image_dictionary.items():
    print(f"Car ID {track_id}: Collected {len(images)} processed images.")

#print("Summary of Collected text")
#for track_id, text in OCR_readings.items():
    #print(f"Car ID {track_id}: Collected {len(text)} processed plates.")

print("last seen frames")
print(last_seen_frame)

#print("OCR readings")
#print(OCR_readings)

print("majority vote")
print(majority_vote)
