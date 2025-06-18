import os
import cv2
import csv
import torch
from model import get_model
from PIL import Image
from torchvision.models import ResNet18_Weights

# Load model, transform, and class labels
weights = ResNet18_Weights.DEFAULT
class_idx = weights.meta["categories"]

model, transform = get_model()

IMAGES_DIR = "images"
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

log_path = os.path.join(OUTPUTS_DIR, "inspection_log.csv")
with open(log_path, 'w', newline='') as log:
    writer = csv.writer(log)
    writer.writerow(["filename", "predicted_class", "confidence"])

    for filename in os.listdir(IMAGES_DIR):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        path = os.path.join(IMAGES_DIR, filename)
        image = Image.open(path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred_class = torch.max(probs, 0)

        class_name = class_idx[pred_class.item()]
        label = f"{class_name}, Conf: {conf.item():.2f}"

        img_cv = cv2.imread(path)
        cv2.putText(img_cv, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUTPUTS_DIR, filename), img_cv)

        writer.writerow([filename, class_name, f"{conf.item():.4f}"])
