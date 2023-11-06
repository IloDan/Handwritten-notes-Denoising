import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import imutils

def deskew_and_crop(image_path):
    # Load the image
    orig_img = cv2.imread(image_path)
    img = orig_img.copy()
 
    new_width = 800
    new_height = 800

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height))

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Blur grayscale image
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Opening morphological transformation with a rectangular kernel
    # kernel maggiori nell'opening rimuovono più rumore che confonde l'algorimto a capire i bordi, più k è piccolo è più smooth sarà l'eliminazione deel rumore e di dettagli
    opening = cv2.morphologyEx(blur_img, cv2.MORPH_OPEN, np.ones((19, 19), np.uint8))

    # Gradient morphological transformation, obtain the outline of the image
    kernel = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)

    # Find edges with Canny
    edges = cv2.Canny(image=gradient, threshold1=75, threshold2=200)

    # Find contours in edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour, which should be the document
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Find the reference points of the document
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Draw reference points on the document
    contour_img = resized_img.copy()
    contour_img = cv2.drawContours(contour_img, [box], 0, (0, 0, 255), 1)

    dst_pts = np.array(box, dtype="float32")
    src_pts = np.array([(0, 0), (resized_img.shape[1]-1, 0), (resized_img.shape[1]-1, resized_img.shape[0]-1), (0, resized_img.shape[0]-1)], dtype="float32")

    # Calculate the transformation matrix
    M = cv2.getPerspectiveTransform(dst_pts, src_pts)


    # Apply perspective correction
    corrected_img = cv2.warpPerspective(resized_img, M, (resized_img.shape[1], resized_img.shape[0]))

    return orig_img, resized_img, gray_img, blur_img, opening, gradient, edges, corrected_img, contour_img

def main():
    image_path = os.path.abspath('./input/test_1.jpg')

    orig_img, resized_img, gray_img, blur_img, opening, gradient, edges, contour_img, corrected_img = deskew_and_crop(image_path)

    # Create a single figure to display all the images
    plt.figure(figsize=(12, 8))

    # Plot the resized image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    plt.title('Starting Image')

    # Plot the grayscale image
    plt.subplot(2, 3, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale Image')

    # Plot the blurred image
    plt.subplot(2, 3, 3)
    plt.imshow(blur_img, cmap='gray')
    plt.title('Blurred grayscale Image')

    # Plot the opening image
    plt.subplot(2, 3, 4)
    plt.imshow(opening, cmap='gray')
    plt.title('Opening Image')

    # Plot the gradient image
    plt.subplot(2, 3, 5)
    plt.imshow(gradient, cmap='gray')
    plt.title('Gradient Image')

    # Plot the edges
    plt.subplot(2, 3, 6)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges with Canny')

    plt.figure(figsize=(12, 8))

    plt.tight_layout()
    plt.show()

    # Plot the corrected image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
    plt.title('Starting Image')

    # Plot the contour image
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title('Dewarped Image')


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
