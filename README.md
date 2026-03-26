# Offroad Semantic Segmentation 🚗🌿

Off-road scene understanding using deep learning. This project uses U-Net with a ResNet34 backbone to perform pixel-wise classification of terrain elements like trees, sky, rocks, and more.

## 🚀 Features
- 10-class semantic segmentation
- Lightweight U-Net (ResNet34 encoder)
- Data augmentation (flip)
- Combined loss (CrossEntropy + Dice)
- Prediction visualization with class percentages
  
## 📂 Dataset
This project uses the Offroad Segmentation Dataset containing labeled images of outdoor environments with 10 semantic classes.

## 🧠 Model Code

```python
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim

# -------------------------
# Classes + Colors
# -------------------------
CLASS_NAMES = {0:"Trees",1:"Lush Bushes",2:"Dry Grass",3:"Dry Bushes",
               4:"Ground Clutter",5:"Flowers",6:"Logs",7:"Rocks",8:"Landscape",9:"Sky"}
NUM_CLASSES = len(CLASS_NAMES)

COLOR_MAP = np.array([
    [0,255,0], [34,139,34], [189,183,107], [160,82,45], [128,128,128],
    [255,192,203], [139,69,19], [105,105,105], [210,180,140], [135,206,235]
])

# -------------------------
# Dataset with augmentations
# -------------------------
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256,256))/255.0

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_NEAREST)

        # -------------------------
        # Tiny augmentations
        # -------------------------
        if np.random.rand() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if np.random.rand() < 0.3:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        image = torch.from_numpy(image).permute(2,0,1).float()

        # -------------------------
        # Mask mapping
        # -------------------------
        mapped_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        mapped_mask[(mask >= 90) & (mask <= 110)] = 0
        mapped_mask[(mask >= 190) & (mask <= 210)] = 1
        mapped_mask[(mask >= 290) & (mask <= 310)] = 2
        mapped_mask[(mask >= 490) & (mask <= 510)] = 3
        mapped_mask[(mask >= 540) & (mask <= 560)] = 4
        mapped_mask[(mask >= 590) & (mask <= 610)] = 5
        mapped_mask[(mask >= 690) & (mask <= 710)] = 6
        mapped_mask[(mask >= 790) & (mask <= 810)] = 7
        mapped_mask[(mask >= 900) & (mask <= 910)] = 8
        mapped_mask[(mask >= 1000) & (mask <= 1010)] = 9
        mask = torch.from_numpy(mapped_mask).long()

        return image, mask

# -------------------------
# Paths + DataLoader
# -------------------------
base_path = "C:/Users/Deekshitha/Downloads/Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/train"
image_dir = os.path.join(base_path, "Color_Images")
mask_dir = os.path.join(base_path, "Segmentation")

image_files = sorted(os.listdir(image_dir))
image_paths, mask_paths = [], []
for file in image_files:
    img_path = os.path.join(image_dir, file)
    mask_path = os.path.join(mask_dir, file)
    if os.path.exists(mask_path):
        image_paths.append(img_path)
        mask_paths.append(mask_path)

# Use all 50 images
image_paths = image_paths[:50]
mask_paths = mask_paths[:50]

dataset = SegmentationDataset(image_paths, mask_paths)
val_split = int(0.2*len(dataset))
train_dataset, val_dataset = random_split(dataset, [len(dataset)-val_split, val_split])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

# -------------------------
# Model
# -------------------------
device = torch.device("cpu")
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=3, classes=NUM_CLASSES).to(device)

# -------------------------
# Loss + Optimizer
# -------------------------
# Boost Sky & rare classes
class_weights = torch.tensor([0.5,5,2,5,3,10,7,5,20,20])
loss_fn = nn.CrossEntropyLoss(weight=class_weights.float())

def dice_loss(pred, target, smooth=1e-5):
    pred = torch.softmax(pred, dim=1)
    target_one_hot = nn.functional.one_hot(target, NUM_CLASSES).permute(0,3,1,2).float()
    intersection = (pred*target_one_hot).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
    dice = 1 - ((2*intersection+smooth)/(union+smooth))
    return dice.mean()

optimizer = optim.Adam(model.parameters(), lr=5e-4)

# -------------------------
# Training for 50 epochs
# -------------------------
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, masks) + dice_loss(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds==masks).sum().item()
            total += masks.numel()
    val_acc = (correct/total)*100

    # Print progress
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

# -------------------------
# Visualization
# -------------------------
model.eval()
with torch.no_grad():
    for i in range(min(5,len(dataset))):
        image, mask = dataset[i]
        input_img = image.unsqueeze(0).to(device)
        output = model(input_img)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        total_pixels = mask.numel()
        print(f"\nImage {i+1} GT Percentages:")
        for class_id,class_name in CLASS_NAMES.items():
            percent = (np.sum(mask.numpy()==class_id)/total_pixels)*100
            print(f"{class_name}: {percent:.2f}%")

        print(f"\nImage {i+1} Pred Percentages:")
        for class_id,class_name in CLASS_NAMES.items():
            percent = (np.sum(pred==class_id)/total_pixels)*100
            print(f"{class_name}: {percent:.2f}%")

        img_np = (image.permute(1,2,0).numpy()*255).astype(np.uint8)
        gt_color = COLOR_MAP[mask.numpy()]
        pred_color = COLOR_MAP[pred]
        combined = np.concatenate([img_np, gt_color, pred_color], axis=1)

        plt.figure(figsize=(12,4))
        plt.imshow(combined)
        plt.title(f"Input | Ground Truth | Prediction (Image {i+1})")
        plt.axis('off')
        plt.show()
```

## ⚙️ Requirements
- Python
- PyTorch
- OpenCV
- segmentation_models_pytorch

## ▶️ How to Run
1. Download dataset
2. Update dataset path in code
3. Run the script

## 📊 Output
<img width="1600" height="611" alt="image" src="https://github.com/user-attachments/assets/90313940-70c2-4d84-bdcf-1a43285cab32" />
Image 1 GT Percentages:
Trees: 90.55%
Lush Bushes: 0.07%
Dry Grass: 6.63%
Dry Bushes: 0.32%
Ground Clutter: 1.56%
Flowers: 0.00%
Logs: 0.00%
Rocks: 0.87%
Landscape: 0.00%
Sky: 0.00%
Image 1 Pred Percentages:
Trees: 85.76%
Lush Bushes: 0.07%
Dry Grass: 9.57%
Dry Bushes: 0.43%
Ground Clutter: 3.55%
Flowers: 0.00%
Logs: 0.00%
Rocks: 0.61%
Landscape: 0.00%
Sky: 0.00%
<img width="1582" height="556" alt="image" src="https://github.com/user-attachments/assets/ca2823d8-005f-4991-89c8-13f7fdf2d274" />
Image 2 GT Percentages:
Trees: 89.38%
Lush Bushes: 0.07%
Dry Grass: 7.65%
Dry Bushes: 0.31%
Ground Clutter: 1.70%
Flowers: 0.00%
Logs: 0.00%
Rocks: 0.90%
Landscape: 0.00%
Sky: 0.00%

Image 2 Pred Percentages:
Trees: 83.85%
Lush Bushes: 0.07%
Dry Grass: 10.76%
Dry Bushes: 0.49%
Ground Clutter: 3.96%
Flowers: 0.00%
Logs: 0.00%
Rocks: 0.87%
Landscape: 0.00%
Sky: 0.00%
<img width="1480" height="531" alt="image" src="https://github.com/user-attachments/assets/c4474ec6-5086-4c20-9529-a5dbfa2a3f6e" />
Image 3 GT Percentages:
Trees: 87.95%
Lush Bushes: 0.08%
Dry Grass: 8.64%
Dry Bushes: 0.52%
Ground Clutter: 1.88%
Flowers: 0.00%
Logs: 0.00%
Rocks: 0.93%
Landscape: 0.00%
Sky: 0.00%

Image 3 Pred Percentages:
Trees: 81.47%
Lush Bushes: 0.08%
Dry Grass: 12.82%
Dry Bushes: 0.74%
Ground Clutter: 3.98%
Flowers: 0.00%
Logs: 0.00%
Rocks: 0.92%
Landscape: 0.00%
Sky: 0.00%
<img width="1600" height="571" alt="image" src="https://github.com/user-attachments/assets/47518b5a-f5b1-47bc-938b-41f50b3d864d" />
Image 4 GT Percentages:
Trees: 86.50%
Lush Bushes: 0.09%
Dry Grass: 9.62%
Dry Bushes: 0.69%
Ground Clutter: 2.05%
Flowers: 0.00%
Logs: 0.00%
Rocks: 1.05%
Landscape: 0.00%
Sky: 0.00%

Image 4 Pred Percentages:
Trees: 79.50%
Lush Bushes: 0.11%
Dry Grass: 13.74%
Dry Bushes: 1.05%
Ground Clutter: 4.52%
Flowers: 0.00%
Logs: 0.00%
Rocks: 1.09%
Landscape: 0.00%
Sky: 0.00%
<img width="1597" height="538" alt="image" src="https://github.com/user-attachments/assets/9912ddbf-f5bf-4722-a5f1-fd07d66743a8" />
Image 5 GT Percentages:
Trees: 83.18%
Lush Bushes: 0.09%
Dry Grass: 11.83%
Dry Bushes: 0.93%
Ground Clutter: 2.75%
Flowers: 0.00%
Logs: 0.00%
Rocks: 1.22%
Landscape: 0.00%
Sky: 0.00%

Image 5 Pred Percentages:
Trees: 73.83%
Lush Bushes: 0.09%
Dry Grass: 18.05%
Dry Bushes: 1.28%
Ground Clutter: 5.34%
Flowers: 0.00%
Logs: 0.00%
Rocks: 1.41%
Landscape: 0.00%
Sky: 0.00%

## 📉 Training Performance

- Trained for 50 epochs  
- Initial Loss: 2.49 → Final Loss: 1.11  
- Best Validation Accuracy: 77.80% (Epoch 42)  
The model shows steady improvement in early epochs and stabilizes after 30~40 epochs, indicating good convergence.

## 📈 Observations
The model performs well on dominant classes like Trees but struggles with minority classes such as Flowers and Logs due to class imbalance. We improved the baseline by incorporating data augmentation, class weighting, and Dice loss, along with a validation split to better evaluate generalization.

## 🚀 Future Work
- Improve performance on minority classes
- Use larger dataset
- Train on GPU for better accuracy

