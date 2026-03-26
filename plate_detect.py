import cv2
from ultralytics import YOLO
import easyocr

# Load NUMBER PLATE model (custom)
model = YOLO("keremberke/yolov8m-license-plate")

reader = easyocr.Reader(['en'])

url = "http://172.20.10.2:8080/video"
cap = cv2.VideoCapture(url)

cv2.namedWindow("Plate Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Plate Detection", 800, 500)

print("Press q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # Crop plate
            plate = frame[y1:y2, x1:x2]

            # OCR
            text_result = reader.readtext(plate)

            for (bbox, text, prob) in text_result:
                if prob > 0.5:
                    text = text.replace(" ", "").upper()
                    print("Plate:", text)

                    cv2.putText(frame, text,
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0,255,255),
                                2)

    cv2.imshow("Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()