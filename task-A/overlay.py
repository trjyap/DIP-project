import cv2
from main import prep_video

def overlay_video(video_file_path, output_file_name):
    # Prepares the video file for processing
    vid,out,total_frames = prep_video(video_file_path, output_file_name)

    # Read overlay video 
    overlay = cv2.VideoCapture(getOverlayPath)
    if not overlay.isOpened():
        print(f"Error: Cannot open video file {overlay_file_path}")
        return None, None, 0

    # Gets the new width and height needed for the overlay
    new_width, new_height = getResolution()

    # Loop through video frames to process overlay
    for frame_count in range(int(total_frames)):
        # Read frames from the main video
        success_main, main_frame = vid.read()
        if not success_main:
            break

        # Read frames from the overlay video
        success_overlay, overlay_frame = overlay.read()
        if not success_overlay:
            # If overlay video ends, restart from the first frame
            overlay.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success_overlay, overlay_frame = overlay.read()
        
        # Resize the overlay frame to the specified resolution
        resized_overlay = cv2.resize(overlay_frame, (new_width, new_height))

        # Determine the position for the overlay (top-right corner)
        main_height, main_width, _ = main_frame.shape
        x_offset = main_width - new_width  # Horizontal offset
        y_offset = 0  # Vertical offset

        # Place the overlay onto the main frame
        overlay_region = main_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width]

        # Blend the overlay with the specified region
        combined_region = cv2.addWeighted(overlay_region, 0.5, resized_overlay, 0.5, 0)
        main_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = combined_region

        # Write the combined frame to the output video
        out.write(main_frame)

    # Release resources
    vid.release()
    overlay.release()
    out.release()
    print("Overlay video created successfully!")
        

def getResolution(): 
    # Function to get the resolution from the user
    resolution = input("Enter the resolution (e.g., 1920x1080): ")
    return tuple(map(int, resolution.split("x")))

def getOverlayPath():
    # Function to get the overlay video path from the user
    overlay_file_path = input("Enter the path of the overlay video: ")
    return overlay_file_path


    