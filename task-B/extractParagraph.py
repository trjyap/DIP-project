import cv2
import numpy as np
import os
import glob

def remove_table(binary_image, table_density_threshold=0.8):
    """
    Detects and removes table regions from a binary image.
    
    Args:
        binary_image: Grayscale binary image (white text on black background).
        table_density_threshold: Fraction of the width that defines a dense row as a table row.

    Returns:
        A tuple (non_table_image, table_mask) where:
            - non_table_image: The binary image with table regions removed.
            - table_mask: Binary mask of the table regions.
    """
    height, width = binary_image.shape

    # Compute horizontal histogram (number of white pixels per row)
    horizontal_histogram = np.sum(binary_image == 255, axis=1)

    # Detect table rows based on high density
    table_rows = np.where(horizontal_histogram > table_density_threshold * width)[0]

    # Create a mask for table regions
    table_mask = np.zeros_like(binary_image)
    if len(table_rows) > 0:
        table_mask[table_rows.min():table_rows.max(), :] = 255

    # Remove table regions from the binary image
    non_table_image = cv2.bitwise_and(binary_image, cv2.bitwise_not(table_mask))

    return non_table_image, table_mask

def column_detection(binary_image, min_column_width=50):
    """
    Detects column boundaries in a binary image.
    
    Args:
        binary_image: Grayscale binary image (white text on black background).
        min_column_width: Minimum width for a valid column.

    Returns:
        List of tuples representing column boundaries (start, end).
    """
    vertical_histogram = np.sum(binary_image == 255, axis=0)
    column_gaps = np.where(vertical_histogram == 0)[0]

    column_boundaries = []
    prev_gap = 0
    for gap in column_gaps:
        if gap - prev_gap > min_column_width:
            column_boundaries.append((prev_gap, gap))
        prev_gap = gap
    if prev_gap < binary_image.shape[1]:
        column_boundaries.append((prev_gap, binary_image.shape[1]))

    return column_boundaries

def paragraph_detection(column_image, max_gap=30, table_mask=None):
    """
    Detects paragraphs within a column image.
    
    Args:
        column_image: Binary image of a single column.
        max_gap: Maximum gap (in rows) to consider as part of the same paragraph.
        table_mask: Optional binary mask of table regions to exclude overlapping paragraphs.

    Returns:
        List of tuples representing paragraph boundaries (top, bottom).
    """
    height, _ = column_image.shape
    horizontal_histogram = np.sum(column_image == 255, axis=1)
    text_rows = np.where(horizontal_histogram > 0)[0]

    paragraphs = []
    current_paragraph = []

    for i in range(len(text_rows)):
        if not current_paragraph:
            current_paragraph.append(text_rows[i])
        else:
            gap = text_rows[i] - text_rows[i - 1]
            if gap <= max_gap:
                current_paragraph.append(text_rows[i])
            else:
                paragraphs.append(current_paragraph)
                current_paragraph = [text_rows[i]]
    if current_paragraph:
        paragraphs.append(current_paragraph)

    refined_paragraphs = []
    for paragraph in paragraphs:
        top = paragraph[0]
        bottom = paragraph[-1]

        if table_mask is not None:
            table_intersection = np.any(table_mask[max(0, top - 30):min(height, bottom + 30), :] == 255)
            if table_intersection:
                continue

        refined_paragraphs.append((max(0, top - 30), min(height, bottom + 30)))

    return refined_paragraphs

def process_images_with_columns(image_paths, output_dir="task-B/paragraphs"):
    """
    Processes images to detect columns and paragraphs, removing tables.

    Args:
        image_paths: List of image file paths to process.
        output_dir: Directory to save the processed paragraph images.
    """
    os.makedirs(output_dir, exist_ok=True)

    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Read and preprocess the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not read the image at {image_path}")
            continue

        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        non_table_image, table_mask = remove_table(binary_image)

        # Detect columns
        column_boundaries = column_detection(non_table_image)

        for col_idx, (start_col, end_col) in enumerate(column_boundaries):
            column_image = non_table_image[:, max(0, start_col - 30):min(non_table_image.shape[1], end_col + 30)]

            # Detect paragraphs within the column
            paragraphs = paragraph_detection(column_image, table_mask=table_mask[:, start_col - 30:end_col + 30])

            # Process each paragraph
            for idx, (top, bottom) in enumerate(paragraphs):
                cropped_paragraph = column_image[top:bottom, :]
                paragraph_height, paragraph_width = cropped_paragraph.shape

                # Check if the paragraph contains a table
                horizontal_histogram = np.sum(cropped_paragraph == 255, axis=1)
                table_row_indices = np.where(horizontal_histogram > 0.8 * paragraph_width)[0]
                if len(table_row_indices) > 0:
                    # Skip the paragraph if it contains a table
                    continue

                # Check if the paragraph contains an image
                image_like_rows = horizontal_histogram > 5
                consecutive_black_rows = 0
                for is_black_row in image_like_rows:
                    if is_black_row:
                        consecutive_black_rows += 2
                        if consecutive_black_rows > 1:
                            # Skip the paragraph if it contains an image
                            continue
                    else:
                        consecutive_black_rows = 0

                # Check if the paragraph is larger than 700x700, this is especially for 004.png
                if paragraph_height > 700 and paragraph_width > 700:
                    # Re-detect columns within this paragraph
                    paragraph_columns = column_detection(cropped_paragraph)

                    for sub_col_idx, (sub_start_col, sub_end_col) in enumerate(paragraph_columns):
                        sub_column_image = cropped_paragraph[:, max(0, sub_start_col - 30):min(paragraph_width, sub_end_col + 30)]

                        # Re-detect paragraphs within the sub-column
                        sub_paragraphs = paragraph_detection(sub_column_image)

                        # Save refined paragraphs
                        for sub_idx, (sub_top, sub_bottom) in enumerate(sub_paragraphs):
                            sub_cropped_paragraph = sub_column_image[sub_top:sub_bottom, :]
                            # Skip paragraphs smaller than 20x20
                            if sub_cropped_paragraph.shape[0] < 40 or sub_cropped_paragraph.shape[1] < 40:
                                continue
                            sub_cropped_paragraph = cv2.bitwise_not(sub_cropped_paragraph)
                            output_path = os.path.join(output_dir, f"{base_name}_column_{sub_col_idx + 1}_paragraph_{sub_idx + 1}.png")
                            cv2.imwrite(output_path, sub_cropped_paragraph)
                else:
                    # Save the paragraph directly if it's not larger than 700x700
                    cropped_paragraph = cv2.bitwise_not(cropped_paragraph)
                    output_path = os.path.join(output_dir, f"{base_name}_column_{col_idx + 1}_paragraph_{idx + 1}.png")
                    cv2.imwrite(output_path, cropped_paragraph)

        print(f"Processed '{image_path}' with {len(column_boundaries)} columns, tables excluded.")

def process_folder_with_images(input_folder, output_dir="task-B/paragraphs"):
    """
    Processes all images in a folder.

    Args:
        input_folder: Folder containing images to process.
        output_dir: Directory to save processed outputs.
    """
    image_paths = glob.glob(os.path.join(input_folder, "*.png"))
    process_images_with_columns(image_paths, output_dir)

# Example usage
process_folder_with_images("task-B/project-files-B", "task-B/paragraphs")