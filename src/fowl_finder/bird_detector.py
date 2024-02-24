from ultralytics import YOLO
from pathlib import Path
import torch

minimum_confidence = 0.4

def find_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def find_bird_key(result):
    return find_key_by_value(result.names, 'bird')

def mid(x1, x2):
    return (x1 + x2) / 2

def tensor_mid(t):
    return (mid(t[0], t[2]), mid(t[1], t[3]))

def image_contains_bird(results, confidence=minimum_confidence):
    for result in results:
        boxes = result.boxes
        classifiers = boxes.cls
        return torch.any(classifiers == find_bird_key(result))

def find_bird(results):
    for result in results:
        boxes = result.boxes
        indices = torch.where(boxes.cls == find_bird_key(result))
        bird_boxes = boxes.xyxy[indices]
        for box in bird_boxes:
            print(tensor_mid(box))

model = YOLO("yolov8n.pt")

sparrow_results = model('resources/sparrows.webp')
raspberry_results = model('resources/raspberry.webp')

print(f"Does the image of just a raspberry have birds: {image_contains_bird(raspberry_results)}")
print(f"Does the image of the sparrows contain any birds: {image_contains_bird(sparrow_results)}")

print("The midpoints of all sparrows are:")
find_bird(sparrow_results)
