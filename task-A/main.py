import numpy as np
import cv2
from matplotlib import pyplot as plt

# Function to import the video file and set output video file
def prep_video(file_path):
    vid = cv2.VideoCapture(file_path)
    
    # Checks if the video capture object was successfully opened
    if not vid.isOpened():
        print(f"Error: Cannot open video file {file_path}")
        return None, None, 0
    
    # Gets the width and height of the frames in the video
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Frame width: {frame_width}, Frame height: {frame_height}")

    # Sets the output video file
    out = cv2.VideoWriter("task-A/processed-files-A/processed_video.avi",
                          cv2.VideoWriter_fourcc(*"MJPG"), 
                          30.0, 
                          (frame_width, frame_height))
    
    # Gets the total number of frames
    total_no_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames: {total_no_frames}")
    
    return vid, out, total_no_frames


# ===========================================================================================================
# MAIN FUNCTION
# ===========================================================================================================

if __name__ == "__main__":
    # Detects night in the video and brightens the video if it is night
    from detectNight import detect_night
    print("Enter path of video to check for night: ")
    file_path = input()
    detect_night(file_path)

# Blurs faces in the video


# Resizes and overlays the talking video on the top left


# Adds watermark to the video


# Appends ending screen to the video