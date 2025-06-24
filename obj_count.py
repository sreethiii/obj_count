import cv2
import numpy as np
from collections import Counter

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use webcam or replace with a video file path

# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Function to perform object detection
def detect_objects(frame):
    height, width, _ = frame.shape

    # Prepare the image for detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Lists to store detection results
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maxima Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Extract the actual indexes (handles both cases)
    if isinstance(indexes, tuple):
        indexes = indexes[0]  # Get the array of indexes

    final_boxes = []
    final_class_ids = []
    for i in indexes.flatten():  # Now safely flatten the indexes array
        final_boxes.append(boxes[i])
        final_class_ids.append(class_ids[i])

    return final_boxes, final_class_ids

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Initialize the object counter for each frame
    object_counter = Counter()

    # Perform object detection
    boxes, class_ids = detect_objects(frame)

    # Count occurrences of each detected object
    detected_classes = [classes[class_id] for class_id in class_ids]
    object_counter.update(detected_classes)

    # Draw bounding boxes and labels on the frame
    for box, class_id in zip(boxes, class_ids):
        x, y, w, h = box
        label = classes[class_id]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display detected objects and their counts
    text = "Detected objects: " + ", ".join([f"{key}: {value}" for key, value in object_counter.items()])
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection with Counting", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
