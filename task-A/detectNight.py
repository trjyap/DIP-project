import numpy as np
import cv2

# Function to detect night in the video
def detect_night(file_path, output_file_name):
    from main import prep_video
    # Prepares the video for processing
    vid, out, total_no_frames = prep_video(file_path, output_file_name)

    # Checks if the video was prepared successfully
    if vid is None or out is None:
        print("Error: Video preparation failed.")
        return
    else: 
        print("Night detection in progress...\n")

    # Initialises total brightness value for calculation
    total_brightness = 0.0

    # Calculates total brightness 
    for frame_count in range(0, int(total_no_frames)):
        # Checks for successful reading of the frame
        success, frame = vid.read() 
        if not success:
            print(f"Error: Cannot read the frame at frame count {frame_count}")
            break

        # Converts frame to grayscale for brightness calculation
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        # Calculates the average brightness of the frame
        avg_brightness = np.mean(gray_frame)
        
        total_brightness += avg_brightness

    # Calculates the average brightness of the video
    vid_avg_brightness = total_brightness / total_no_frames
    print(f"Total brightness of the video: {total_brightness}")
    print(f"Average brightness of the video: {vid_avg_brightness}")

    # Prints out detection results
    if vid_avg_brightness < 100:
        print("Night detected in the video. Increasing brightness...\n")
    else:
        print("No night detected in the video. No brightness adjustment needed.\n")

    # Reset the video capture to the beginning
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Apply brightness adjustment frame by frame
    for frame_count in range(0, int(total_no_frames)):
        success, frame = vid.read()
        if not success:
            print(f"Error: Cannot read the frame at frame count {frame_count}")
            break
        
        # Multiplies pixels by 1.35 if video brightness is below 100
        if vid_avg_brightness < 100:
            frame = cv2.convertScaleAbs(frame, alpha=1.35, beta=0)
        else:
            frame = cv2.convertScaleAbs(frame, alpha=1, beta=0)

        # Writes the frame into the output video file
        out.write(frame)

    # Releases the video for other use
    vid.release()
    out.release()
    cv2.destroyAllWindows()
    print("Night detection and brightness adjustment complete. Video released.\n")