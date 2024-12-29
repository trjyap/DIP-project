import cv2

# Function to import the video file and set output video file
def prep_video(file_path, output_file_name):
    vid = cv2.VideoCapture(file_path)
    
    # Checks if the video capture object was successfully opened
    if not vid.isOpened():
        print(f"Error: Cannot open video file {file_path}")
        return None, None, 0
    
    # Gets the width and height of the frames in the video
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Frame width: {frame_width}, Frame height: {frame_height}")

    # Sets a constant output path for processed videos
    output_path = "task-A/processed-files-A/"

    # Joins the output file name with the output path
    full_output_path = output_path + output_file_name

    # Checks if the file already exists and modifies the name if necessary
    base_name, ext = full_output_path.rsplit('.', 1)
    counter = 1
    while True:
        try:
            with open(full_output_path, 'x'):
                break
        except FileExistsError:
            print("File name already exists. Modifying...")
            full_output_path = f"{base_name}_{counter}.{ext}"
            print(f"New file name: {base_name}_{counter}.{ext}")
            counter += 1

    # Sets the output video file
    out = cv2.VideoWriter(full_output_path,
                          cv2.VideoWriter_fourcc(*"MJPG"), 
                          30.0, 
                          (frame_width, frame_height))
    
    # Gets the total number of frames
    total_no_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames: {total_no_frames}")
    
    return vid, out, total_no_frames

# Gets user input for video path and output file name
def get_user_input():
    file_path = input("Enter path of video to process: ")
    output_file_name = input("Enter the output file name (use .avi extension): ")
    return file_path, output_file_name


# ===========================================================================================================
# MAIN FUNCTION
# ===========================================================================================================

def main():
    while True:
        print("\n============= SIMPLE VIDEO PROCESSING TOOL =============")
        print("Select a task to perform:")
        print("1. Detect night and brighten the video")
        print("2. Blur faces in the video")
        print("3. Resize and overlay the talking video on the top left")
        print("4. Add watermark to the video")
        print("5. Append ending screen to the video")
        print("6. Exit")
        choice = input("Enter your choice (1-6): ")

        if choice == '6':
            print("Exiting... Goodbye!\n")
            break
        # Detects night in the video and brightens the video if it is night
        elif choice == '1':     
            file_path, output_file_name = get_user_input()
            from detectNight import detect_night
            detect_night(file_path, output_file_name)
        # Blurs faces in the video
        elif choice == '2':
            file_path, output_file_name = get_user_input()
            from blurFaces import blur_faces
            blur_faces(file_path, output_file_name)
        # Resizes and overlays the talking video on the top left
        elif choice == '3':
            file_path, output_file_name = get_user_input()
            from overlay import overlay_video(file_path, output_file_name)
            overlay_video(file_path, output_file_name)
        # Adds watermark to the video
        # elif choice == '4':
        #     file_path, output_file_name = get_user_input()
        #     add_watermark(file_path, output_file_name)
        # Appends ending screen to the video (or another video to stitch)
        elif choice == '5':
            file_path, output_file_name = get_user_input()
            from stitching import stitching
            stitching(file_path, output_file_name)
        else:
            print("Invalid choice. Please select a valid option.")

# Ensures that main() runs only when it is called directly
if __name__ == "__main__":
    main()