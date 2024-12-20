import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def extract_paragraphs_with_histograms(image_paths, horizontal_thresh=10, vertical_thresh=10, min_area_threshold=1000, output_dir="output", histogram_dir="histogram"):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    os.makedirs(histogram_dir, exist_ok=True)  # Ensure the histogram directory exists

    for image_path in image_paths:
        # Load the image as is (black text on white background)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image {image_path}")
            continue

        # Convert to grayscale for histogram processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binary thresholding (keeping black text on white background)
        _, binary_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Calculate horizontal and vertical histograms
        horizontal_hist = np.sum(binary_img == 0, axis=1)  # Count black pixels per row
        vertical_hist = np.sum(binary_img == 0, axis=0)    # Count black pixels per column

        # Save histograms as plots
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        plt.figure(figsize=(12, 6))

        # Horizontal Histogram
        plt.subplot(2, 1, 1)
        plt.plot(horizontal_hist, color='blue')
        plt.title(f"Horizontal Histogram for {base_name}")
        plt.xlabel("Row Index")
        plt.ylabel("Black Pixel Count")

        # Vertical Histogram
        plt.subplot(2, 1, 2)
        plt.plot(vertical_hist, color='green')
        plt.title(f"Vertical Histogram for {base_name}")
        plt.xlabel("Column Index")
        plt.ylabel("Black Pixel Count")

        # Save the histogram as an image
        histogram_output_path = os.path.join(histogram_dir, f"{base_name}_histogram.png")
        plt.tight_layout()
        plt.savefig(histogram_output_path)  # Save the plot to a file
        plt.close()  # Close the plot to free memory
        print(f"Saved histogram for {base_name} at {histogram_output_path}")

        # Extract ranges based on horizontal and vertical histograms
        horizontal_mask = horizontal_hist > horizontal_thresh
        vertical_mask = vertical_hist > vertical_thresh

        row_ranges = extract_ranges(horizontal_mask)
        col_ranges = extract_ranges(vertical_mask)

        paragraph_count = 1
        for row_start, row_end in row_ranges:
            for col_start, col_end in col_ranges:
                # Extract the paragraph region
                region = binary_img[row_start:row_end, col_start:col_end]

                # Filter out small areas
                if region.shape[0] * region.shape[1] < min_area_threshold:
                    continue

                # Save the paragraph image
                paragraph_output_path = os.path.join(output_dir, f"{base_name}_paragraph_{paragraph_count}.png")
                cv2.imwrite(paragraph_output_path, region)
                paragraph_count += 1

        print(f"Extracted paragraphs for {base_name} saved to {output_dir}")

def extract_ranges(mask):
    """
    Extract start and end ranges from a 1D boolean mask.

    Args:
        mask (np.array): A 1D boolean array indicating presence.

    Returns:
        list of tuples: Start and end indices for detected regions.
    """
    ranges = []
    start = None

    for i, val in enumerate(mask.tolist()):
        if val and start is None:  # Start of a new range
            start = i
        elif not val and start is not None:  # End of a range
            ranges.append((start, i))
            start = None
    if start is not None:  # Add the last range if still open
        ranges.append((start, len(mask)))

    return ranges

# List of input images
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

# Process the images
extract_paragraphs_with_histograms(image_paths, horizontal_thresh=10, vertical_thresh=10, min_area_threshold=1000)
