import cv2
from main import prep_video

def add_watermark(video_path, output_file_name):

    vid,out,total_frames = prep_video(video_file_path, output_file_name)
    watermark = cv2.imread(enter_watermark, cv2.IMREAD_UNCHANGED)
    if watermark is None:
        print("Error: Cannot open watermark file")
        return
    
    # Ensure alpha channel is present, to ensure the watermark blends smoothly
    if watermark.shape[2] == 4:  
        overlay = watermark[:, :, :3]
        mask = watermark[:, :, 3]
    else:
        overlay = watermark
        mask = None

    # Loop through each frame of the video
    for frame_index in range(total_frames):
        ret, frame = vid.read()
        if not ret:
            print(f"Finished processing {frame_index} frames.")
            break
        
        # Apply the full-size watermark to the frame
        if mask is not None:
            for c in range(0, 3):  # Blend each color channel
                frame[:, :, c] = frame[:, :, c] * (1 - mask / 255.0) + overlay[:, :, c] * (mask / 255.0)
        else:
            frame = overlay  # Direct replacement if no transparency

        # Write the watermarked frame to the output video
        out.write(frame)

    # Release resources
    vid.release()
    out.release()
    print(f"Watermarked video saved as {output_file_name}.")