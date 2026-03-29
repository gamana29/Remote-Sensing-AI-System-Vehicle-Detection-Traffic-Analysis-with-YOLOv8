from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

results = model("sample_images/stthomas-parking-transportation-lot.jpg", show=True)

img = results[0].plot()
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

