import cv2
import time
from ultralytics import YOLO
import supervision as sv

model = YOLO("yolov8n.pt")
class_names = model.names

v1 = cv2.VideoCapture("videos/vd1.mp4")

ret, frame0 = v1.read()

H, W = frame0.shape[:2]

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

line_zone = sv.LineZone(
    start = sv.Point(0, int(H * 0.75)),
    end = sv.Point(W, int(H * 0.75))
)

line_annotator = sv.LineZoneAnnotator()

prev_time = 0
prev_pos = {}

while True:
    ret, frame = v1.read()

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.5, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])

    line_zone.trigger(detections)

    anchors = detections.get_anchors_coordinates(
        anchor=sv.Position.BOTTOM_CENTER
    )

    speed_labels = []
    for tracker_id, anchor in zip(detections.tracker_id, anchors):
        current_x, current_y = anchor
        if tracker_id in prev_pos:
            prev_x, prev_y = prev_pos[tracker_id]
            dist = ((current_x - prev_x) ** 2 + (current_y - prev_y) ** 2) ** 0.5
            speed_labels.append(f"{int(dist)} px/frame")
        else:
            speed_labels.append("Calculating")
        prev_pos[tracker_id] = (current_x, current_y)

    class_ids = detections.class_id
    vehicle_class = [
        class_names[class_id]
        for class_id in class_ids
    ]

    car_count = vehicle_class.count("car")
    truck_count = vehicle_class.count("truck")

    labels = [
        f"ID {tracker_id} | {confidence: .2f} | {speed} | {vehicle_type}"
        for tracker_id, confidence, speed, vehicle_type in zip(detections.tracker_id, detections.confidence, speed_labels, vehicle_class)
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
    1,
    (0, 255, 0),
    2
    )

    cv2.putText(
        frame,
        f"Trucks: {truck_count}",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame, f"FPS: {fps:.2f}", (30,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    cv2.imshow("Traffic Analytics System", frame)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
    

v1.release()
cv2.destroyAllWindows()

