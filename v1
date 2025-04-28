
import cv2
import numpy as np

  
# # #
# pip install opencv-python numpy
# # # 


  
def detect_lanes(frame):
    # Griye çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gürültü azalt
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kenar tespiti (Canny)
    edges = cv2.Canny(blur, 50, 150)

    # ROI (Region of Interest) maskeleme
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height*0.6)),
        (0, int(height*0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    # Hough dönüşümüyle çizgi tespiti
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=100,
        maxLineGap=50
    )

    # Çizgileri çiz
    line_image = np.zeros_like(frame)
    if lines is not None:
for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # Orijinal görüntü ile birleştir
    combined = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combined

cap = cv2.VideoCapture("your_video.mp4")  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    lane_frame = detect_lanes(frame)
    cv2.imshow("Lane Detection", lane_frame)

    if cv2.waitKey(1) == 27:  # ESC tuşu
        break

cap.release()
cv2.destroyAllWindows()
