import cv2
import sys
import numpy as np 
from collections import Counter
from skimage import exposure
import splines
from typing import Tuple

def detect_tearing(image: np.ndarray) -> int:
    """
    Detect tearing in an image by identifying horizontal lines that 
    may indicate a tear using edge detection and Hough Line Transformation.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (Canny)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Use Hough Line Transformation to detect strong horizontal lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, 
                            minLineLength=100, maxLineGap=10)

    # Collect horizontal lines and their positions
    if lines is not None:
        horizontal_lines = [(x[0][1], x[0][3]) for x in lines if x[0][1] == x[0][3]]
        if horizontal_lines:
            # Find the most common y-position of horizontal lines
            number = Counter(horizontal_lines)
            most_common_line_y = number.most_common(1)[0][0][0]
            return most_common_line_y
    return None

def split_image_and_reverse(image: np.ndarray, line):
    """
    Split the image into two parts along the detected tear (horizontal line).
    """
    if line is not None and 0 < line < image.shape[0]:
        top_half = image[:int(line), :]
        bottom_half = image[int(line):, :]
        return bottom_half, top_half
    return None, None

def apply_color_correction(image: np.ndarray, true_color: tuple, reference_point: tuple):
    y, x, _ = image.shape

    aberrant_color = image[int(reference_point[1]%y), int(reference_point[0]%x)].astype(float)

    print("Colore errato:", aberrant_color)
    print("Colore reale:", true_color)

    array_adjustment = (true_color / aberrant_color).astype(float)
    print("Fattori di correzzione:", array_adjustment)

    corrected_image = image.astype(float)

    for i in range(corrected_image.shape[0]):
        for j in range(corrected_image.shape[1]):
            corrected_image[i][j] *= array_adjustment
  
    print("Colore corretto:", corrected_image[int(reference_point[1]%y), int(reference_point[0]%x)], end="\n\n")
    return corrected_image.astype(np.uint8)

def normalize_image(image: np.ndarray, max_val: int, min_val: int) -> np.ndarray:
    p_min = np.min(image)
    p_max = np.max(image)
    normalized_image = (image - p_min) / (p_max - p_min) * (max_val-min_val) + min_val
    return normalized_image.astype(np.uint8)

def apply_color_correction_v2(image: np.array, points: Tuple[Tuple[int, int]], true_colors: Tuple[Tuple[int, int, int]]) -> np.array:
    #getting colors from image
    colors = [image[point[1]][point[0]] for point in points]

    #extracting channels
    b_values = [color[0] for color in colors]
    g_values = [color[1] for color in colors]
    r_values = [color[2] for color in colors]

    #extracting channels true colors
    b_true_values = [true_color[0] for true_color in true_colors]
    g_true_values = [true_color[1] for true_color in true_colors]
    r_true_values = [true_color[2] for true_color in true_colors]

    res_b = splines.CatmullRom((0,*b_true_values,255), (0,*b_values,255))
    res_g = splines.CatmullRom((0,*g_true_values,255), (0,*g_values,255))
    res_r = splines.CatmullRom((0,*r_true_values,255), (0,*r_values,255))

    # Make LUT (Lookup Table) from spline
    LUT_b = np.uint8(res_b.evaluate(range(0,256)))
    LUT_g = np.uint8(res_g.evaluate(range(0,256)))
    LUT_r = np.uint8(res_r.evaluate(range(0,256)))

    LUT = np.dstack((LUT_b, LUT_g, LUT_r))

    return cv2.LUT(image, LUT)

def main():
    # Load the image
    img = cv2.imread("./corrupted.png")
    y, x, _ = img.shape
    if img is None:
        sys.exit("Could not read the image.")

    # Detect tearing in the image
    tear_line = detect_tearing(img)
    if tear_line is None:
        sys.exit("Could not detect tearing line in the image.")

    # Split the image at the detected tear line
    top_half, bottom_half = split_image_and_reverse(img, tear_line)
    if top_half is None or bottom_half is None:
        sys.exit("Image splitting failed.")

    matched = exposure.match_histograms(top_half, bottom_half, channel_axis=-1)

    # Define true color references and reference points
    true_colors = ((250, 158,  3),(40, 195, 240))

    points = ((128, 541),(564, 267))

    shifted_image = np.vstack((matched, bottom_half))
    shifted_image = normalize_image(shifted_image, 200, 50)

    #corrected_image = apply_color_correction(shifted_image, true_colors[0], points[0])

    normalized_image = normalize_image(shifted_image, 200, 50)
    cv2.imshow("Shifted & Normalized image", normalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hsv = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([10, 100, 20])
    upper_orange = np.array([25, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Use Canny edge detection
    edges = cv2.Canny(mask, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect shapes that resemble a cone
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:  # Triangle approximation
            cv2.drawContours(normalized_image, [approx], 0, (0, 255, 0), 5)

    # Show the result
    cv2.imshow('Cone Detection', normalized_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()