import glob
import cv2
import numpy as np
import json

# Load models and set mean values
gender_proto = "/content/drive/MyDrive/Opencv_Models/drive-download-20240601T135523Z-001/gender_deploy.prototxt"
gender_model = "/content/drive/MyDrive/Opencv_Models/drive-download-20240601T135523Z-001/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Load gender model
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

def detect_gender(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    return gender

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
video_paths = glob.glob("/content/drive/MyDrive/samples/*.avi")

output_data = []

for video_path in video_paths:
    activity = video_path.split("_")[1]
    print(f"Processing video: {video_path}")
    
    metadata = extract_metadata(video_path)
    print(f"Metadata: {metadata}")
    
    keyframes = extract_keyframes(video_path)
    print(f"Extracted {len(keyframes)} keyframes")
    
    genders = set()
    indoor_outdoor_scores = {"indoors": 0, "outdoors": 0}
    
    for keyframe in keyframes:
        gender = detect_gender(keyframe)
        if gender:
            genders.add(gender)
        classification = classify_indoor_outdoor(keyframe)
        indoor_outdoor_scores[classification] += 1

    print(f"Genders detected: {genders}")
    final_classification = "indoors" if indoor_outdoor_scores["indoors"] > indoor_outdoor_scores["outdoors"] else "outdoors"
    print(f"Video is likely shot {final_classification}")

    video_data = {
        'file_name': video_path.split('/')[-1],
        'activity': activity,
        'metadata': metadata,
        'genders': list(genders),
        'classification': final_classification
    }

    output_data.append(video_data)

print("Processing complete.")

# Save output to JSON file
with open('/content/drive/MyDrive/output_data.json', 'w') as json_file:
    json.dump(output_data, json_file, indent=4)
