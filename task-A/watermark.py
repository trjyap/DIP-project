import cv2
import numpy as np
from main import prep_video

def add_watermark(video_path, output_file_name):
    # Prepare the video file for processing
    vid, out, total_frames = prep_video(video_path, output_file_name)

    # Load the watermark image
    watermark_path = enter_watermark()
    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    if watermark is None:
        print("Error: Cannot open watermark file")
        return

    # Ensure watermark has an alpha channel, or create one
    if watermark.shape[2] == 4:
        overlay = watermark[:, :, :3]  # RGB channels
        alpha = watermark[:, :, 3]    # Alpha channel
    else:
        overlay = watermark
        # Create an alpha mask: make black (0,0,0) fully transparent, others opaque
        alpha = cv2.inRange(overlay, (0, 0, 0), (0, 0, 0))  # Treat non-black as opaque
        alpha = cv2.bitwise_not(alpha)  # Invert so black becomes transparent

    # Loop through each frame of the video
    for frame_index in range(total_frames):
        ret, frame = vid.read()
        if not ret:
            print(f"Finished processing {frame_index} frames.")
            break

        # Get frame and watermark dimensions
        frame_height, frame_width = frame.shape[:2]
        overlay_height, overlay_width = overlay.shape[:2]

        # Ensure the watermark fits within the video frame
        if overlay_height > frame_height or overlay_width > frame_width:
            print("Error: Watermark is larger than the video frame.")
            break

        # Determine position for the watermark (top-left by default)
        x_offset, y_offset = 0, 0

        # Define the region of interest (ROI) on the frame
        roi = frame[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width]

        # Combine the watermark with the frame using the alpha channel
        for c in range(3):  # Apply to each color channel (B, G, R)
            roi[:, :, c] = roi[:, :, c] * (1 - alpha / 255.0) + overlay[:, :, c] * (alpha / 255.0)

        # Replace the ROI in the frame with the combined result
        frame[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width] = roi

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    vid.release()
    out.release()
    print(f"Watermarked video saved as {output_file_name}.")

def enter_watermark():
    return input("Enter the path of the watermark image: ")
