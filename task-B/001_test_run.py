import cv2
import numpy as np
import os
import glob  # For handling multiple files

def process_images(image_paths, output_dir="task-B/paragraphs"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not read the image at {image_path}")
            continue

        # Threshold the image (white text on black background for easier processing)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        # Get the height and width of the image
        height, width = binary_image.shape

        # Compute horizontal histogram (sum of white pixels per row)
        horizontal_histogram = np.sum(binary_image == 255, axis=1)

        # Identify rows with black pixels (text rows)
        text_rows = np.where(horizontal_histogram > 0)[0]

        # Identify paragraph start and end rows
        paragraphs = []
        current_paragraph = []
        max_gap = 30  # Max gap (in rows) to consider as part of the same paragraph

        for i in range(len(text_rows)):
            if not current_paragraph:
                # Start a new paragraph
                current_paragraph.append(text_rows[i])
            else:
                # Check the gap between the current and previous row
                gap = text_rows[i] - text_rows[i-1]
                if gap <= max_gap:
                    current_paragraph.append(text_rows[i])
                else:
                    # Save the completed paragraph and start a new one
                    paragraphs.append(current_paragraph)
                    current_paragraph = [text_rows[i]]
        # Add the last paragraph if it exists
        if current_paragraph:
            paragraphs.append(current_paragraph)

        # Create output images for each paragraph
        output_images = []
        for paragraph in paragraphs:
            top = paragraph[0]
            bottom = paragraph[-1]
            cropped_paragraph = binary_image[max(0, top - 20):min(height, bottom + 20), :]

            # Invert to white text on a black background
            cropped_paragraph = cv2.bitwise_not(cropped_paragraph)
            output_images.append(cropped_paragraph)

        # Save the paragraphs to disk
        for idx, paragraph_image in enumerate(output_images):
            output_path = os.path.join(output_dir, f"{base_name}_paragraph_{idx + 1}.png")
            cv2.imwrite(output_path, paragraph_image)

    print(f"Processing complete. Paragraphs saved to '{output_dir}'.")

# Process multiple images in a folder
def process_folder(input_folder, output_dir="task-B/paragraphs"):
    # Get all PNG files in the folder
    image_paths = [
        "task-B/project-files-B/001.png",
        "task-B/project-files-B/002.png",
        "task-B/project-files-B/003.png",
        "task-B/project-files-B/004.png",
        "task-B/project-files-B/005.png",
        "task-B/project-files-B/006.png",
        "task-B/project-files-B/007.png",
        "task-B/project-files-B/008.png",
    ]
    process_images(image_paths, output_dir)

# Example usage
process_folder("task-B/project-files-B", "task-B/paragraphs")
