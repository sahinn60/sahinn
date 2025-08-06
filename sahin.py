import cv2
import numpy as np
import os

def enhance_low_light_regions(image):
    # Convert to grayscale to detect low-light areas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to identify dark regions (tweak threshold if needed)
    _, low_light_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Smooth image to reduce noise
    smoothed = cv2.GaussianBlur(image, (5, 5), 0)

    # Histogram equalization on brightness channel
    ycrcb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    hist_eq_img = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    # Sharpening kernel
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    sharpened = cv2.filter2D(hist_eq_img, -1, kernel_sharp)

    # Convert mask to 3 channels
    mask_3ch = cv2.merge([low_light_mask] * 3)

    # Apply enhancement only to dark areas
    enhanced = np.where(mask_3ch == 255, sharpened, image)

    return enhanced, low_light_mask

def main():
    print("📷 Low-Light Photo Enhancer")
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

    print("✅ Enhancement complete. Press any key to close the images.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
