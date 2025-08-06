import cv2
import numpy as np
import os

def enhance_low_light_regions(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Enhances low-light regions of an image using histogram equalization
    and sharpening, applied selectively to dark areas.
    
    :param image: Original BGR image
    :return: (enhanced image, low-light mask)
    """
    # Convert to grayscale to detect dark regions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create mask for low-light areas
    _, low_light_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Smooth the image to reduce noise
    smoothed = cv2.GaussianBlur(image, (5, 5), 0)

    # Histogram equalization in Y channel
    ycrcb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    hist_eq_img = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    # Sharpen the image
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    sharpened = cv2.filter2D(hist_eq_img, -1, kernel_sharp)

    # Expand mask to 3 channels
    mask_3ch = cv2.merge([low_light_mask] * 3)

    # Merge enhanced areas only where mask is white (low light)
    enhanced = np.where(mask_3ch == 255, sharpened, image)

    return enhanced, low_light_mask


def main() -> None:
    print("📷 Low-Light Photo Enhancer (Python 3.10 Compatible)")
    path = input("Enter the full path to the image: ").strip()

    if not os.path.isfile(path):
        print("❌ File not found! Please check the path.")
        return

    image = cv2.imread(path)
    if image is None:
        print("❌ Could not read the image. Make sure it's a valid image file.")
        return

    enhanced_img, mask = enhance_low_light_regions(image)

    # Display results
    cv2.imshow("Original Image", image)
    cv2.imshow("Low-Light Mask", mask)
    cv2.imshow("Enhanced Image", enhanced_img)

    print("✅ Enhancement complete. Press any key on image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
