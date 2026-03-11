import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ADE20KDataset, get_transforms
from model import SegmentationViT

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    with tqdm(total=len(dataloader), desc="Training") as pbar:
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
            
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    image_transform, target_transform = get_transforms(image_size=224)
    
    # Ensure './ADE20K' points to your extracted dataset directory
    train_dataset = ADE20KDataset(root='./ADE20K', split='training', 
                                  transform=image_transform, target_transform=target_transform)
    val_dataset = ADE20KDataset(root='./ADE20K', split='validation', 
                                transform=image_transform, target_transform=target_transform)
                                
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # 151 classes (150 semantic labels + 1 background)
    model = SegmentationViT(num_classes=151).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # ignore_index=0 if 0 represents background/unlabeled pixels in ADE20K
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"vit_segmentation_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()