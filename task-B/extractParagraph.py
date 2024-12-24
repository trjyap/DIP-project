import cv2
import numpy as np
import os
import glob

def process_images_with_columns(image_paths, output_dir="task-B/paragraphs"):
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

        # Compute horizontal and vertical histograms
        horizontal_histogram = np.sum(binary_image == 255, axis=1)
        vertical_histogram = np.sum(binary_image == 255, axis=0)

        # Detect tables based on high-density regions in histograms
        table_rows = np.where(horizontal_histogram > 0.8 * width)[0]

        # Create a mask for table regions
        table_mask = np.zeros_like(binary_image)
        if len(table_rows) > 0:
            table_mask[table_rows.min():table_rows.max(),:] = 255

        # Remove table regions from the binary image
        non_table_image = cv2.bitwise_and(binary_image, cv2.bitwise_not(table_mask))

        # Compute vertical histogram (sum of black pixels per column) for non-table regions
        vertical_histogram = np.sum(non_table_image == 255, axis=0)

        # Identify column boundaries based on gaps in the vertical histogram
        column_gaps = np.where(vertical_histogram == 0)[0]
        column_boundaries = []
        prev_gap = 0
        for gap in column_gaps:
            if gap - prev_gap > 50:  # Minimum width for a column (adjustable)
                column_boundaries.append((prev_gap, gap))
            prev_gap = gap
        if prev_gap < width:
            column_boundaries.append((prev_gap, width))  # Add the last column

        # Process each detected column
        column_count = len(column_boundaries)
        for col_idx, (start_col, end_col) in enumerate(column_boundaries):
            # Crop the column region
            column_image = non_table_image[:, start_col - 30 : end_col + 30]

            # Compute horizontal histogram for this column
            horizontal_histogram = np.sum(column_image == 255, axis=1)

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
                cropped_paragraph = column_image[max(0, top - 30):min(height, bottom + 30), :]

                # Invert to white text on a black background
                cropped_paragraph = cv2.bitwise_not(cropped_paragraph)
                output_images.append(cropped_paragraph)

            # Save the paragraphs to disk
            for idx, paragraph_image in enumerate(output_images):
                output_path = os.path.join(output_dir, f"{base_name}_column_{col_idx + 1}_paragraph_{idx + 1}.png")
                cv2.imwrite(output_path, paragraph_image)

        print(f"Processed '{image_path}' with {column_count} columns.")

# Process multiple images in a folder
def process_folder_with_columns(input_folder, output_dir="task-B/paragraphs"):
    # Get all PNG files in the folder
    image_paths = glob.glob(os.path.join(input_folder, "*.png"))
    process_images_with_columns(image_paths, output_dir)

# Example usage
process_folder_with_columns("task-B/project-files-B", "task-B/paragraphs")
