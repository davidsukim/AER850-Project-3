import cv2
import numpy as np
import matplotlib.pyplot as plt

def mask_motherboard(image_path):
    # 1. Loading Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: CANNOT FIND THE IMAGE, CHECK FILE THE LOCATION.")
        return

    original = img.copy()
    
    # 2. preprocess (grayscale & blur)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Edge Detection (Canny)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1) 

    # 4. Contour Detection [cite: 30]
    # only find the outline --> RETR_EXTERNAL
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Find biggest contour (removing small noise)
    if not contours:
        print("Contour could be found.")
        return
    
    # Assuming the largest contour should be the main board
    main_contour = max(contours, key=cv2.contourArea)

    # 6. generating Mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [main_contour], -1, (255), thickness=cv2.FILLED)

    # 7. Using "Bitwise AND" to remove the background
    extracted = cv2.bitwise_and(original, original, mask=mask)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.title("Original")
    
    # Convert BGR into RGB
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 3, 2)
    plt.title("Edges (Canny)")
    plt.imshow(edges, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Extracted PCB")
    plt.imshow(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB))
    
    plt.show()

    # Save the result
    cv2.imwrite("extracted_motherboard.jpg", extracted)
    print("extracted image has been saved as 'extracted_motherboard.jpg'")

# Load (Image path)
mask_motherboard('Project 3 Data/Project 3 Data/motherboard_image.JPEG')