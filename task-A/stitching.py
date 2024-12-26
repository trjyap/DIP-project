import cv2

# Function to stitch a video to other videos
def stitching(file_path, output_file_name):
    from main import prep_video
    # Prepares the video for processing
    vid, out, total_no_frames = prep_video(file_path, output_file_name)

    # Checks if the video was prepared successfully
    if vid is None or out is None:
        print("Error: Video preparation failed.")
        return
    
    # Gets endscreen file path (or another video to stitch)
    stitch_file_path = input("Enter path of video to stitch: ")
    stitch_vid = cv2.VideoCapture(stitch_file_path)

    # Checks if the endscreen video was opened successfully
    if not stitch_vid.isOpened():
        print(f"Error: Cannot open video file {stitch_file_path}")
        return
    else: 
        print("Stitching in progress...\n")
    
    # Writes frames from the main video to the output file
    for frame_count in range(0, int(total_no_frames)):
        success, frame = vid.read()
        if not success:
            print(f"Error: Cannot read the frame at frame count {frame_count}")
            break
        
        # Writes the frame into the output video file
        out.write(frame)

    # Writes frames from the endscreen video to the output file
    while True:
        success, frame = stitch_vid.read()
        if not success:
            break  # Break if no frames are left
        
        # Writes the frame into the output video file
        out.write(frame)

    # Releases the video for other use
    vid.release()
    out.release()
    cv2.destroyAllWindows()
    print("Stitching complete. Video released.\n")