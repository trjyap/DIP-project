import cv2
import numpy as np

# Removes table regions from a binary image based on pixel density analysis
def remove_table(binary_image, table_density_threshold=0.8):
    height, width = binary_image.shape

    # Count white pixels in each row to identify dense regions (likely tables)
    horizontal_histogram = np.sum(binary_image == 255, axis=1)

    # Identify rows that exceed the density threshold (probable table rows)
    table_rows = np.where(horizontal_histogram > table_density_threshold * width)[0]

    # Create a mask marking table regions
    table_mask = np.zeros_like(binary_image)
    if len(table_rows) > 0:
        # Mark the entire region between first and last table row
        table_mask[table_rows.min():table_rows.max(), :] = 255

    # Remove table regions from the original image
    non_table_image = cv2.bitwise_and(binary_image, cv2.bitwise_not(table_mask))
    return non_table_image, table_mask

# Detects text columns in a binary image based on vertical whitespace
def column_detection(binary_image, min_column_width=50):
    # Count white pixels in each column
    vertical_histogram = np.sum(binary_image == 255, axis=0)
    
    # Find completely empty vertical spaces (potential column separators)
    column_gaps = np.where(vertical_histogram == 0)[0]

    # Identify column boundaries based on gaps
    column_boundaries = []
    prev_gap = 0
    for gap in column_gaps:
        if gap - prev_gap > min_column_width:  # Ensure minimum column width
            column_boundaries.append((prev_gap, gap))
        prev_gap = gap
    
    # Add the last column if it extends to the image edge
    if prev_gap < binary_image.shape[1]:
        column_boundaries.append((prev_gap, binary_image.shape[1]))
    return column_boundaries

# Detects paragraphs within a column by analyzing vertical spacing between text lines
def paragraph_detection(column_image, max_gap=30, table_mask=None):
    height, _ = column_image.shape
    # Count white pixels in each row to find text lines
    horizontal_histogram = np.sum(column_image == 255, axis=1)
    text_rows = np.where(horizontal_histogram > 0)[0]
    # Group text rows into paragraphs based on vertical gaps
    paragraphs = []
    current_paragraph = []

    # Build paragraphs by analyzing gaps between consecutive text rows
    for i in range(len(text_rows)):
        if not current_paragraph:
            current_paragraph.append(text_rows[i])
        else:
            gap = text_rows[i] - text_rows[i - 1]
            # If lines close enough to be in same paragraph
            if gap <= max_gap:
                current_paragraph.append(text_rows[i])
            # Gap too large, start new paragraph
            else:
                paragraphs.append(current_paragraph)
                current_paragraph = [text_rows[i]]

    # Refine paragraph boundaries and filter out table regions
    if current_paragraph:
        paragraphs.append(current_paragraph)
    refined_paragraphs = []
    for paragraph in paragraphs:
        top = paragraph[0]
        bottom = paragraph[-1]
        # Skip paragraphs that intersect with tables
        if table_mask is not None:
            table_intersection = np.any(table_mask[max(0, top - 30):min(height, bottom + 30), :] == 255)
            if table_intersection:
                continue

        # Add padding to paragraph boundaries
        refined_paragraphs.append((max(0, top - 30), min(height, bottom + 30)))
    return refined_paragraphs

# Main function to process multiple images, detecting and extracting paragraphs from columns.
def extract_paragraphs(image_paths, output_dir="task-B/paragraphs"):
    for image_path in image_paths:
        base_name = image_path.split('/')[-1].split('.')[0]

        # Load and binarize the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not read the image at {image_path}")
            continue

        # Convert to binary image (white text on black background)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Remove table regions
        non_table_image, table_mask = remove_table(binary_image)

        # Detect columns in the image
        column_boundaries = column_detection(non_table_image)

        # Process each detected column
        for col_index, (start_col, end_col) in enumerate(column_boundaries):
            # Extract column with padding
            column_image = non_table_image[:, max(0, start_col - 30):min(non_table_image.shape[1], end_col + 30)]

            # Detect paragraphs within the column
            paragraphs = paragraph_detection(column_image, table_mask=table_mask[:, start_col - 30:end_col + 30])

            # Process each paragraph in the column
            for index, (top, bottom) in enumerate(paragraphs):
                cropped_paragraph = column_image[top:bottom, :]
                paragraph_height, paragraph_width = cropped_paragraph.shape

                # Ignore paragraphs that appear to be tables
                horizontal_histogram = np.sum(cropped_paragraph == 255, axis=1)
                table_row_indices = np.where(horizontal_histogram > 0.8 * paragraph_width)[0]
                if len(table_row_indices) > 0:
                    continue

                # Special handling for very large paragraphs 700x700 (potentially consist of more than one paragraphs)
                if paragraph_height > 700 and paragraph_width > 700:
                    # Detect columns in the large paragraph
                    paragraph_columns = column_detection(cropped_paragraph)
                    # Process each detected sub-columns
                    for sub_col_index, (sub_start_col, sub_end_col) in enumerate(paragraph_columns):
                        sub_column_image = cropped_paragraph[:, max(0, sub_start_col - 30):min(paragraph_width, sub_end_col + 30)]
                        # Detect paragraphs within the sub-column
                        sub_paragraphs = paragraph_detection(sub_column_image)
                        # Process each paragraph in the sub-column
                        for sub_index, (sub_top, sub_bottom) in enumerate(sub_paragraphs):
                            sub_cropped_paragraph = sub_column_image[sub_top:sub_bottom, :]
                            # Ignore very small paragraphs 40x40 (probably just a full stop image)
                            if sub_cropped_paragraph.shape[0] < 40 or sub_cropped_paragraph.shape[1] < 40:
                                continue
                                
                            # Invert colors for output (black text on white background)
                            sub_cropped_paragraph = cv2.bitwise_not(sub_cropped_paragraph)
                            output_path = f"{output_dir}/{base_name}_column_{sub_col_index + 1}_paragraph_{sub_index + 1}.png"
                            cv2.imwrite(output_path, sub_cropped_paragraph)
                else:
                    # Process normal-sized paragraphs (paragraphs smaller than 700x700)
                    cropped_paragraph = cv2.bitwise_not(cropped_paragraph)
                    output_path = f"{output_dir}/{base_name}_column_{col_index + 1}_paragraph_{index + 1}.png"
                    cv2.imwrite(output_path, cropped_paragraph)
        # Print which image have how many columns detected
        print(f"Processed {base_name} with {len(column_boundaries) - 1} columns detected.")

# Process images
def process_images(output_dir="task-B/paragraphs"):
    image_paths = [
        "task-B/project-files-B/001.png",
        "task-B/project-files-B/002.png",
        "task-B/project-files-B/003.png",
        "task-B/project-files-B/004.png",
        "task-B/project-files-B/005.png",
        "task-B/project-files-B/006.png",
        "task-B/project-files-B/007.png",
        "task-B/project-files-B/008.png"
        ]
    extract_paragraphs(image_paths, output_dir)

# Run the main function to process images and extract paragraphs from columns
process_images()
