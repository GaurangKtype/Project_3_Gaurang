import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def thresholding(imgPath):
    img = cv.imread(imgPath)
    imgGray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    hist=cv.calcHist([imgGray], [0], None, [256], [0,256])
    plt.figure()
    plt.plot(hist)
    plt.xlabel('bins')
    plt.ylabel('# of pixel')

    thresOpt = [cv.THRESH_BINARY,
                cv.THRESH_BINARY_INV,
                cv.THRESH_TOZERO,
                cv.THRESH_TOZERO_INV,
                cv.THRESH_TRUNC]
    
    thresNames = ['binary', 'binaryInv', 'toZero', 'toZeroInv', 'trunc']
    
    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(imgGray, cmap='gray')
    
    for i in range(len(thresOpt)):
        plt.subplot(2, 3, i+2)  
        _, imgThres = cv.threshold(imgGray,60, 255, thresOpt[i])
        plt.imshow(imgThres, cmap='gray')
        plt.title(thresNames[i])
    
    plt.tight_layout()
    plt.show()

def callback(x):
    pass

def cannyEdge(imgPath):
    img = cv.imread(imgPath)
    if img is None:
        print("Error: Image not found")
        return
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgGray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    height, width, _ = img.shape
    scale = 1 / 5
    heightScale = int(height * scale)
    widthScale = int(width * scale)
    img = cv.resize(img, (widthScale, heightScale), interpolation=cv.INTER_LINEAR)
    
    winname = 'canny'
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.createTrackbar('minThres', winname, 0, 255, callback)
    cv.createTrackbar('maxThres', winname, 0, 255, callback)    
    
    while True:
        minThres = cv.getTrackbarPos('minThres', winname)
        maxThres = cv.getTrackbarPos('maxThres', winname)
        cannyEdge = cv.Canny(imgGray, minThres, maxThres)
        cv.imshow(winname, cannyEdge)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
    
def harrisEdge(imgPath):
    img = cv.imread(imgPath)
    if img is None:
        print("Error: Image not found")
        return

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    winname = 'Harris Corner Detection'
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.createTrackbar('blockSize', winname, 2, 10, lambda x: x)
    cv.createTrackbar('kSize', winname, 3, 31, lambda x: x)

    while True:
        blockSize = cv.getTrackbarPos('blockSize', winname)
        kSize = cv.getTrackbarPos('kSize', winname)

        # Ensure kSize is odd and blockSize is greater than 1
        if kSize % 2 == 0:
            kSize += 1
        if blockSize < 2:
            blockSize = 2

        harris = cv.cornerHarris(imgGray, blockSize, kSize, 0.04)
        imgDisplay = img.copy()
        imgDisplay[harris > 0.01 * harris.max()] = [0, 0, 255]

        cv.imshow(winname, imgDisplay)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()


def contour(imgPath):
    img = cv.imread(imgPath)
    
    # Convert the image to RGB and grayscale
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    
    # Apply a binary threshold to the grayscale image
    _, img_binary = cv.threshold(img_gray, 70, 255, cv.THRESH_BINARY_INV)
    
    # Find contours on the binary image
    contours, _ = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    # Draw contours on a copy of the original image
    img_contours = img_rgb.copy()
    cv.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
    
    # Prepare to display the images in a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(15, 20))
    
    # Actual Image
    axs[0, 0].imshow(img_rgb)
    axs[0, 0].set_title('Actual Image')
    axs[0, 0].axis('off')
    
    # Gray-scale Image
    axs[0, 1].imshow(img_gray, cmap='gray')
    axs[0, 1].set_title('Gray-scale Image')
    axs[0, 1].axis('off')
    
    # Binary Image
    axs[1, 0].imshow(img_binary, cmap='gray')
    axs[1, 0].set_title('Binary Image')
    axs[1, 0].axis('off')
    
    # Binary Image with Contours
    axs[1, 1].imshow(img_contours)
    axs[1, 1].set_title('Binary Image with Contours')
    axs[1, 1].axis('off')
    
    # Display the number of contours found
    print("Number of contours found:", len(contours))
    
    # Save the contour figure
    #contour_image_path = '/mnt/data/motherboard_contours.png'
    #plt.savefig(contour_image_path)

    plt.show()
    return contours

def find_motherboard_contour(contours):
    largest_area = 0
    largest_contour_index = -1
    contour_rect = None  # This will store the rectangle boundary of the largest contour

    for index, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area > largest_area:
            # Approximate contour to polygon
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * peri, True)

            # Check if the approximated contour has 4 sides (likely to be a rectangle)
            if len(approx) == 4:
                largest_area = area
                largest_contour_index = index
                contour_rect = cv.boundingRect(approx)

    return largest_contour_index, contour_rect
    
def draw_and_display_contour_by_index(img_path, contour_index):
    # Read the image from the given path
    img = cv.imread(img_path)
    
    # Convert the image to RGB for displaying
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Convert the image to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to the grayscale image
    _, img_binary = cv.threshold(img_gray, 70, 255, cv.THRESH_BINARY_INV)
    
    # Find contours on the binary image
    contours, _ = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Check if the contour index is valid
    if contour_index < 0 or contour_index >= len(contours):
        raise ValueError("Contour index out of range.")
    
    # Draw the specified contour on a copy of the original image
    img_with_contour = img_rgb.copy()
    cv.drawContours(img_with_contour, [contours[contour_index]], -1, (0, 255, 0), 3)
    
    # Display the image with the specified contour
    plt.imshow(img_with_contour)
    plt.title(f'Image with Contour at Index {contour_index}')
    plt.axis('off')
    plt.show()
    
def mask_largest_contour(img, contours):
    largest_contour = sorted(contours, key=cv.contourArea, reverse=True)[0]
    mask = np.zeros_like(img)
    cv.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv.FILLED)
    masked_img = cv.bitwise_and(img, mask)
    return masked_img
    
def find_second_largest_contour(contours):
    # Sort the contours based on contour area and then reverse the sorted array
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
    
    # Skip the first largest contour and return the second largest contour
    if len(sorted_contours) > 1:
        return sorted_contours[1]
    else:
        return None  # If there is no second contour

def mask_motherboard(img, contour):
    # Create a mask with the same dimensions as the image, initially black (all zeros)
    mask = np.zeros_like(img)
    
    # Draw the contour on the mask with white color and filled in
    cv.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv.FILLED)
    
    # Apply the mask using bitwise_and
    masked_img = cv.bitwise_and(img, mask)
    
    return masked_img

    
img_path = 'G:\My Drive\Gaurang Files\TMU\Year 4\AER 850 Intro to Machine Learning\Project\Project_3_Gaurang\motherboard_image.JPEG'  

img = cv.imread(img_path)

# Convert to grayscale and threshold to create a binary image
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, img_binary = cv.threshold(img_gray, 70, 255, cv.THRESH_BINARY_INV)

# Find contours on the binary image
contours, _ = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# Mask the largest contour
masked_largest_contour = mask_largest_contour(img, contours)

# Display the masked largest contour
plt.imshow(cv.cvtColor(masked_largest_contour, cv.COLOR_BGR2RGB))
plt.title('Largest Contour Masked Out')
plt.axis('off')
plt.show()

# Find the second largest contour
second_largest_contour = find_second_largest_contour(contours)

# Mask the motherboard using the second largest contour
masked_motherboard = mask_motherboard(img, second_largest_contour)

# Convert the masked image to RGB for displaying with matplotlib
masked_motherboard_rgb = cv.cvtColor(masked_motherboard, cv.COLOR_BGR2RGB)

# Display the result
plt.imshow(masked_motherboard_rgb)
plt.title('Motherboard Masked Out')
plt.axis('off')
plt.show()

contour(img_path)
#cannyEdge(img_path)
#harrisEdge(img_path)
