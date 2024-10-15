import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.notebook import tqdm

class StabilityDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, augment=False):
        self.stability_data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.stability_data) * (2 if self.augment else 1)

    def __getitem__(self, idx):
        original_idx = idx // 2 if self.augment else idx
        flip = self.augment and idx % 2 == 1

        img_name = str(self.stability_data.iloc[original_idx, 0])
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        
        image = Image.open(img_path).convert('RGB')
        
        stability_height = self.stability_data.iloc[original_idx, -1]
        stability_class = int(stability_height) - 1

        if flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(stability_class, dtype=torch.long)

class StabilityPredictor(nn.Module):
    def __init__(self, num_classes=6):
        super(StabilityPredictor, self).__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.efficientnet = models.efficientnet_v2_s(weights=weights)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model = None
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_training=True)
        
        # Validation phase
        model.eval()
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, is_training=False)
        
        # Learning rate scheduler step
        scheduler.step(val_loss)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 60)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            model.load_state_dict(best_model)
            break

    return model

def run_epoch(model, data_loader, criterion, optimizer, device, is_training=True):
    running_loss = 0.0
    correct = 0
    total = 0

    # Create progress bar
    progress_bar = tqdm(data_loader, desc="Training" if is_training else "Validating")

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        if is_training:
            optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        if is_training:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

# Load pre-calculated dataset statistics
stats = torch.load('dataset_stats.pth')
mean, std = stats['mean'], stats['std']
print(f"Loaded dataset mean: {mean}")
print(f"Loaded dataset std: {std}")

# Create transform with loaded normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
])

# Create full dataset with augmentation and correct normalization
full_dataset = StabilityDataset(csv_file='./COMP90086_2024_Project_train/train.csv', 
                                img_dir='./COMP90086_2024_Project_train/train', 
                                transform=transform,
                                augment=True)  # Enable augmentation

# Split dataset into train and validation
val_ratio = 0.025
dataset_size = len(full_dataset)
val_size = int(val_ratio * dataset_size)
train_size = dataset_size - val_size
print(f'Splitting dataset into {(1 - val_ratio)}:{val_ratio} training/test split')
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

model = StabilityPredictor(num_classes=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

print('Training...')
model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, patience=3)

torch.save(model.state_dict(), 'stability_predictor_efficientnetv2_classification_augmented.pth')

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    image_ids = []

    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy() + 1)  # Add 1 to convert back to 1-6 range
            image_ids.extend(ids.numpy())  # Convert tensor to numpy array

    return predictions, image_ids
    
# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pre-calculated dataset statistics
stats = torch.load('dataset_stats.pth')
mean, std = stats['mean'], stats['std']
print(f"Loaded dataset mean: {mean}")
print(f"Loaded dataset std: {std}")

# Create transform with loaded normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
])

# Load the unlabeled dataset
test_dataset = StabilityDataset(csv_file='./COMP90086_2024_Project_test/test.csv', 
                                img_dir='./COMP90086_2024_Project_test/test', 
                                transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load the trained model
model = StabilityPredictor(num_classes=6)
model.load_state_dict(torch.load('stability_predictor_efficientnetv2_classification_augmented.pth'))
model.to(device)

# Make predictions
predictions, image_ids = predict(model, test_loader, device)

# Save predictions to CSV
with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'labels'])
    for img_id, pred in zip(image_ids, predictions):
        writer.writerow([int(img_id) + 1, int(pred)])  # Ensure both are integers

print("Predictions saved to predictions.csv")
