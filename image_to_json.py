import cv2
import numpy as np
import json

EPSILON = 0.01

dataset = []

def get_matrix_of_image(image, c):
    height, width = image.shape
    newX, newY = image.shape[1] * 28.0 / width, image.shape[0] * 28.0 / height
    small_image = cv2.resize(image,(int(newX + EPSILON),int(newY + EPSILON)))
    ret = [[0 for i in range(28)] for j in range(28)]
    for i in range(28):
        for j in range(28):
            if int(small_image[i][j]) >= 10:
                ret[i][j] = 1
            else:
                ret[i][j] = 0
    return ret

image_file_names = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg"]

for image_file_name in image_file_names:

    print(image_file_name)

    image  = cv2.imread(image_file_name)

    height, width, depth = image.shape
    resize_scale = 0.85
    newX,newY = image.shape[1]*resize_scale, image.shape[0]*resize_scale
    image = cv2.resize(image,(int(newX),int(newY)))
    height, width, depth = image.shape

    # cv2.imshow("Image", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)

    median = cv2.medianBlur(gray, 7)

    blur = cv2.GaussianBlur(median, (5,5), 0)
    # cv2.imshow("blur", blur)

    thresh = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    # cv2.imshow("thresh", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    c = 0
    for i in contours:
            area = cv2.contourArea(i)
            if area > height * width / 2:
                if area > max_area:
                    max_area = area
                    best_cnt = i
                    # image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
            c+=1

    mask = np.zeros((gray.shape),np.uint8)
    cv2.drawContours(mask,[best_cnt],0,255,-1)
    cv2.drawContours(mask,[best_cnt],0,0,2)
    # cv2.imshow("mask", mask)

    out = np.zeros_like(gray)
    out[mask == 255] = gray[mask == 255]
    # cv2.imshow("New image", out)

    median = cv2.medianBlur(out, 7)
    # median = blur
    # cv2.imshow("median1", median)

    blur = cv2.GaussianBlur(median, (5,5), 0)
    # blur = cv2.medianBlur(blur, 7)
    # blur = out
    # cv2.imshow("blur1", blur)

    # _, thresh = cv2.threshold(median, 60, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    # cv2.imshow("thresh1", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1] + cv2.boundingRect(ctr)[0] * image.shape[0])
    contours = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[0] ** 2 + cv2.boundingRect(ctr)[1] ** 2) ** 0.5 + cv2.boundingRect(ctr)[0] * width * 2)

    image_without_contours = image.copy()

    c = 0
    j = 0
    last_y = cv2.boundingRect(contours[0])[0]
    digit_number = 0

    for i in contours:
            image_with_contour = image.copy()
            area = cv2.contourArea(i)
            if area > height * width / 500 and area < height * width / 120:
                j += 1
                # print(j, digit_number)
                y = cv2.boundingRect(i)[0]
                diff_abs = y - last_y
                if diff_abs < 0:
                    diff_abs = -diff_abs
                if diff_abs > width / 20:
                    digit_number += 1
                last_y = cv2.boundingRect(i)[0]
                mask = np.zeros_like(image_with_contour)
                cv2.drawContours(mask, contours, c, 255, -1)
                out = np.zeros_like(image_with_contour) # Extract out the object and place into output image
                out[mask == 255] = image_with_contour[mask == 255]

                (y, x, _) = np.where(mask == 255)
                (topy, topx) = (np.min(y), np.min(x))
                (bottomy, bottomx) = (np.max(y), np.max(x))
                cropped_image = thresh[topy+35:bottomy-34, topx+35:bottomx-34]
                mat = get_matrix_of_image(cropped_image, c)
                # cv2.imshow("Cropped Image " + str(c), cropped_image)
                current_digit = {"digit": j, "pixels": mat}
                dataset.append(current_digit)
            c+=1


with open("dataset.json", "w") as f:
    json.dump(dataset, f)

# cv2.imshow("Final Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
