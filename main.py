import cv2
from ultralytics import YOLO
import supervision as sv

model = YOLO("yolov8s.pt")
class_names = model.names

v1 = cv2.VideoCapture("videos/vd1.mp4")

ret, frame0 = v1.read()

H, W = frame0.shape[:2]

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

line_zone = sv.LineZone(
        start = sv.Point(0, H//2),
        end = sv.Point(W, H//2)
    )

line_annotator = sv.LineZoneAnnotator()

while True:
    ret, frame = v1.read()

    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.5, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])

    line_zone.trigger(detections)

    class_ids = detections.class_id

    vehicle_class = [
        class_names[class_id]
        for class_id in class_ids
    ]

    car_count = vehicle_class.count("car")
    truck_count = vehicle_class.count("truck")

    labels = [
        f"ID {tracker_id} | {confidence: .2f}"
        for tracker_id, confidence in zip(detections.tracker_id, detections.confidence)
    ]

    frame = box_annotator.annotate(
        scene=frame,
        detections=detections
    )

    frame = label_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels,
    )

    frame = line_annotator.annotate(
        frame=frame,
        line_counter=line_zone
    )

    cv2.putText(
    frame,
    f"Cars: {car_count}",
    (30, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
    )

    cv2.putText(
        frame,
        f"Trucks: {truck_count}",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Traffic Analytics System", frame)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
    

v1.release()
cv2.destroyAllWindows()

