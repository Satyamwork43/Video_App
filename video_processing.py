import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import numpy as np

# Load models and set mean values
face_proto = "/content/drive/MyDrive/Opencv_Models/drive-download-20240601T135523Z-001/opencv_face_detector.pbtxt"
face_model = "/content/drive/MyDrive/Opencv_Models/drive-download-20240601T135523Z-001/opencv_face_detector_uint8.pb"
gender_proto = "/content/drive/MyDrive/Opencv_Models/drive-download-20240601T135523Z-001/gender_deploy.prototxt"
gender_model = "/content/drive/MyDrive/Opencv_Models/drive-download-20240601T135523Z-001/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Load models
face_net = cv2.dnn.readNet(face_model, face_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# Gender categories
GENDER_LIST = ['Male', 'Female']

def extract_metadata(video_path):
    video = cv2.VideoCapture(video_path)
    metadata = {
        'duration': int(video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)),
        'resolution': (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        'fps': video.get(cv2.CAP_PROP_FPS)
    }
    video.release()
    return metadata

def extract_keyframes(video_path, threshold=30, keyframe_interval=1):
    video = cv2.VideoCapture(video_path)
    keyframes = []
    prev_frame = None
    frame_count = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if prev_frame is None:
            prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keyframes.append(frame)
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_frame, gray_frame)
        non_zero_count = np.count_nonzero(diff)

        if non_zero_count > threshold * 1000 and frame_count % keyframe_interval == 0:
            keyframes.append(frame)

        prev_frame = gray_frame
        frame_count += 1

    video.release()
    return keyframes

def detect_humans_and_gender(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    genders = set()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frame.shape[1])
            y1 = int(detections[0, 0, i, 4] * frame.shape[0])
            x2 = int(detections[0, 0, i, 5] * frame.shape[1])
            y2 = int(detections[0, 0, i, 6] * frame.shape[0])

            face = frame[y1:y2, x1:x2]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]
            genders.add(gender)
    
    return list(genders)

def classify_indoor_outdoor(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    blue_mask = cv2.inRange(hsv, np.array([100, 40, 40]), np.array([140, 255, 255]))

    green_pixels = np.sum(green_mask > 0)
    blue_pixels = np.sum(blue_mask > 0)
    total_pixels = frame.shape[0] * frame.shape[1]

    if (green_pixels + blue_pixels) / total_pixels > 0.1:
        return "outdoors"
    else:
        return "indoors"

# Process a collection of videos
video_paths = ['/content/v_Archery_g15_c01.avi']  # Add more video paths as needed
for video_path in video_paths:
    print(f"Processing video: {video_path}")
    
    metadata = extract_metadata(video_path)
    print(f"Metadata: {metadata}")
    
    keyframes = extract_keyframes(video_path)
    print(f"Extracted {len(keyframes)} keyframes")
    
    genders = set()
    indoor_outdoor_scores = {"indoors": 0, "outdoors": 0}
    
    for keyframe in keyframes:
        genders.update(detect_humans_and_gender(keyframe))
        classification = classify_indoor_outdoor(keyframe)
        indoor_outdoor_scores[classification] += 1

    print(f"Genders detected: {genders}")
    final_classification = "indoors" if indoor_outdoor_scores["indoors"] > indoor_outdoor_scores["outdoors"] else "outdoors"
    print(f"Video is likely shot {final_classification}")

    # Display keyframes (optional)
    for keyframe in keyframes:
        cv2_imshow(keyframe)

print("Processing complete.")
