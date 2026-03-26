import cv2
from ultralytics import YOLO
import easyocr

# Load models
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'])

url = "http://172.20.10.2:8080/video"
cap = cv2.VideoCapture(url)

print("Press q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ["car", "motorbike", "bus", "truck"]:

                # Draw vehicle box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                # Crop region (vehicle area)
                crop = frame[y1:y2, x1:x2]

                # OCR
                result = reader.readtext(crop)

                for (bbox, text, prob) in result:
                    if prob > 0.5:
                        print("Detected:", text)

                        cv2.putText(frame, text,
                                    (x1, y1-30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0,255,255),
                                    2)

    cv2.imshow("Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()