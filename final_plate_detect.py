import cv2
from ultralytics import YOLO
import easyocr
import re

# 🔥 Load your trained plate model
model = YOLO("plate_model.pt")

# OCR reader
reader = easyocr.Reader(['en'])

# Camera (choose one)

# 👉 Mobile camera
url = "http://172.20.10.3:8080/video"
cap = cv2.VideoCapture(url)

# 👉 OR Laptop webcam (better performance)
# cap = cv2.VideoCapture(0)

# Window
cv2.namedWindow("Plate Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Plate Detection", 800, 500)

print("Press 'q' to exit")

frame_count = 0

while True:
    frame_count += 1

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 🔥 Resize frame (reduce lag)
    frame = cv2.resize(frame, (640, 480))

    # 🔥 Skip frames for performance
    if frame_count % 3 != 0:
        cv2.imshow("Plate Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Detect plates
    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # Crop plate
            plate = frame[y1:y2, x1:x2]

            # 🔥 Preprocessing (VERY IMPORTANT)
            plate = cv2.resize(plate, None, fx=2, fy=2)
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            # OCR
            ocr_results = reader.readtext(thresh)

            final_text = ""

            for (bbox, text, prob) in ocr_results:
                text = text.replace(" ", "").upper()
                text = re.sub(r"[^A-Z0-9]", "", text)

                if prob > 0.4:
                    final_text += text

            print("Combined OCR:", final_text)

            # 🔥 HSRP pattern (India)
            pattern = r"[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}"

            if re.search(pattern, final_text):
                plate_number = re.search(pattern, final_text).group()
                print("✅ HSRP Plate:", plate_number)

                cv2.putText(frame, plate_number,
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,255),
                            2)

    cv2.imshow("Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()