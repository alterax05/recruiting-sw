import cv2
import sys
import numpy as np
from collections import Counter

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

def split_image(image: np.ndarray, line):
    """
    Split the image into two parts along the detected tear (horizontal line).
    """
    if line is not None and 0 < line < image.shape[0]:
        top_half = image[:int(line), :]
        bottom_half = image[int(line):, :]
        return bottom_half, top_half
    return None, None

def apply_color_correction(image, true_color, reference_point):
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
    top_half, bottom_half = split_image(img, tear_line)
    if top_half is None or bottom_half is None:
        sys.exit("Image splitting failed.")

    # Define true color references and reference points
    true_color1 = np.array([250, 158,  3], dtype=float) #point1
    true_color2 = np.array([40, 195, 240], dtype=float) #point2
    
    point1 = (128, 541)
    point2 = (564, 267)

    shifted_image = np.vstack((top_half, bottom_half))

    cv2.imshow("Shifted image", shifted_image)
    pointed_image = cv2.line(shifted_image, point1, point1, true_color1, 10)
    pointed_image = cv2.line(pointed_image, point2, point2, true_color2, 5)
    cv2.imshow("Points",pointed_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    corrected_top = apply_color_correction(top_half, true_color2, point2)
    corrected_bottom = apply_color_correction(bottom_half, true_color1, point1)
    cv2.imshow("Shifted Image", pointed_image)
    cv2.imshow("Corrected Image", np.vstack((corrected_top, corrected_bottom)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()