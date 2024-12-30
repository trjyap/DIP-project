import cv2
from main import prep_video

def overlay_video(video_file_path, output_file_name):
    # Prepares the video file for processing
    vid, out, total_frames = prep_video(video_file_path, output_file_name)

    # Read overlay video
    overlay_path = getOverlayPath()
    overlay = cv2.VideoCapture(overlay_path)
    if not overlay.isOpened():
        print(f"Error: Cannot open video file {overlay_path}")
        return

    # Gets the new width and height needed for the overlay
    new_width, new_height = getResolution()

    # Loop through video frames using a for loop
    for frame_count in range(int(total_frames)):
        # Read frames from the main video
        success_main, main_frame = vid.read()
        if not success_main:
            print(f"Stopped: Unable to read frame {frame_count} from the main video.")
            break

        # Read frames from the overlay video
        success_overlay, overlay_frame = overlay.read()
        if not success_overlay:
            # If overlay video ends, restart from the first frame
            overlay.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success_overlay, overlay_frame = overlay.read()
            if not success_overlay:
                print(f"Error: Unable to read frame {frame_count} from the overlay video after restart.")
                break

        # Resize the overlay frame to the specified resolution
        resized_overlay = cv2.resize(overlay_frame, (new_width, new_height))

        # Determine the position for the overlay (top-left corner)
        main_height, main_width, _ = main_frame.shape
        x_offset = 0  # Horizontal offset (top-left)
        y_offset = 0  # Vertical offset (top-left)

        # Ensure the overlay fits within the main frame
        if y_offset + new_height > main_height or x_offset + new_width > main_width:
            print("Error: Overlay size exceeds main frame dimensions.")
            break

        # Place the overlay directly on the main frame
        main_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_overlay

        # Write the combined frame to the output video
        out.write(main_frame)

    # Release resources
    vid.release()
    overlay.release()
    out.release()
    print(f"Overlay video created successfully! Processed {frame_count + 1} frames.")

def getResolution(): 
    # Function to get the resolution from the user
    resolution = input("Enter the resolution (e.g., 1920x1080): ")
    return tuple(map(int, resolution.split("x")))

def getOverlayPath():
    # Function to get the overlay video path from the user
    overlay_file_path = input("Enter the path of the overlay video: ")
    return overlay_file_path
