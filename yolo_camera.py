import cv2
from ultralytics import YOLO

# Load YOLO model (downloads automatically first time)
model = YOLO("yolov8n.pt")

# Your mobile camera URL
url = "http://172.20.10.2:8080/video"

cap = cv2.VideoCapture(url)
cv2.namedWindow("YOLO Vehicle Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Vehicle Detection", 800, 500)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Camera started... Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame)

    # Draw bounding boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = model.names[cls]

            # Detect only vehicles
            if label in ["car", "motorbike", "bus", "truck"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)

    # Show output
    cv2.imshow("YOLO Vehicle Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()