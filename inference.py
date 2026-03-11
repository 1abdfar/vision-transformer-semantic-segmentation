import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import SegmentationViT
from dataset import get_transforms

def predict_and_visualize(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = SegmentationViT(num_classes=151).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Prepare Image
    image_transform, _ = get_transforms(image_size=224)
    original_image = Image.open(image_path).convert("RGB")
    input_tensor = image_transform(original_image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    # Visualize
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    # Resize original image to match tensor dimensions for display
    plt.imshow(original_image.resize((224, 224)))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Segmentation Map")
    plt.imshow(prediction, cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage:
    # predict_and_visualize("path/to/test_image.jpg", "vit_segmentation_epoch_10.pth")
    pass