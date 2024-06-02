import streamlit as st
import json
import base64
import os

# Function to display video
def display_video(video_path):
    try:
        video_file = open(video_path, 'rb')
        video_bytes = video_file.read()
        b64_encoded_video = base64.b64encode(video_bytes).decode('utf-8')
        video_url = f'data:video/mp4;base64,{b64_encoded_video}'
        st.video(video_url)
    except FileNotFoundError:
        st.warning(f"Video file {video_path} not found. Skipping this file.")

# Load the output data from the JSON file
try:
    with open('output_data.json', 'r') as json_file:
        videos_data = json.load(json_file)
except FileNotFoundError:
    st.error("JSON file not found. Please ensure 'output_data.json' is in the correct directory.")
    st.stop()
except json.JSONDecodeError:
    st.error("Error decoding JSON file. Please check the file format.")
    st.stop()

# Create filters in the sidebar
st.sidebar.header('Filter Videos')
activity_type = st.sidebar.selectbox('Activity Type', ['All'] + sorted(set(video['activity'] for video in videos_data)))
gender = st.sidebar.selectbox('Gender', ['All'] + sorted(set(g for video in videos_data for g in video['genders'])))
shot_location = st.sidebar.selectbox('Shot Location', ['All', 'indoors', 'outdoors'])

# Filter videos based on user input
filtered_videos = [
    video for video in videos_data
    if (activity_type == 'All' or video['activity'] == activity_type) and
       (gender == 'All' or gender in video['genders']) and
       (shot_location == 'All' or video['classification'] == shot_location)
]

st.header('Filtered Videos')

if filtered_videos:
    displayed_videos = 0
    for video in filtered_videos:
        if displayed_videos >= 5:
            break
        st.subheader(f"Video: {video['file_name']}")
        st.write(f"Activity: {video['activity']}")
        st.write(f"Gender: {', '.join(video['genders'])}")
        st.write(f"Shot Location: {video['classification']}")
        display_video(video['file_name'])  # Assuming the video file is in the same directory
        displayed_videos += 1
else:
    st.write("No videos match the selected criteria.")
