import cv2
import numpy as np

# # #
# pip install opencv-python numpy
# # #

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([[
        (0, height),
        (img.shape[1], height),
        (img.shape[1], int(height * 0.6)),
        (0, int(height * 0.6))
    ]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def color_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Beyaz çizgi aralığı
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Sarı çizgi aralığı
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    filtered = cv2.bitwise_and(img, img, mask=combined_mask)
    return filtered

def perspective_transform(img):
    height, width = img.shape[:2]
    src = np.float32([
        [width*0.43, height*0.65],
        [width*0.58, height*0.65],
        [width*0.1, height],
        [width*0.95, height]
    ])
    dst = np.float32([
        [width*0.2, 0],
        [width*0.8, 0],
        [width*0.2, height],
        [width*0.8, height]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (width, height))

def detect_lanes(frame):
    filtered = color_filter(frame)
    birdseye = perspective_transform(filtere…
combined = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return combined

Video oynatma
cap = cv2.VideoCapture("your_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    lanes = detect_lanes(frame)
    cv2.imshow("Advanced Lane Detection", lanes)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
