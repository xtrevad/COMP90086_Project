import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models
from torchvision.models import EfficientNet_V2_S_Weights, convnext_base
from PIL import Image
import pandas as pd
import numpy as np
import os
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.notebook import tqdm
import platform
import multiprocessing


class StabilityDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, augment=False, image_size=224):
        self.stability_data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        self.image_size = image_size
        self.augmented_indices = self._create_augmented_indices() if augment else None

    def _create_augmented_indices(self):
        base_indices = list(range(len(self.stability_data)))
        flipped_indices = [idx + len(self.stability_data) for idx in base_indices]
        zoomed_indices = [idx + 2 * len(self.stability_data) for idx in base_indices]
        zoomed_flipped_indices = [idx + 3 * len(self.stability_data) for idx in base_indices]
        return base_indices + flipped_indices + zoomed_indices + zoomed_flipped_indices

    def __len__(self):
        return len(self.stability_data) * 4 if self.augment else len(self.stability_data)

    def __getitem__(self, idx):
        if self.augment:
            original_idx = idx % len(self.stability_data)
            augmentation = idx // len(self.stability_data)
        else:
            original_idx = idx
            augmentation = 0

        img_name = str(self.stability_data.iloc[original_idx, 0])
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        
        image = Image.open(img_path).convert('RGB')
        
        stability_height = self.stability_data.iloc[original_idx, -1]
        stability_class = int(stability_height) - 1

        if self.augment:
            if augmentation in [1, 3]:  # Flip
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if augmentation in [2, 3]:  # Zoom
                width, height = image.size
                crop_size = int(min(width, height) * 0.8)  # Zoom in by 20%
                left = (width - crop_size) // 2
                top = (height - crop_size) // 2
                right = left + crop_size
                bottom = top + crop_size
                image = image.crop((left, top, right, bottom))

        # Resize the image to ensure consistent size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(stability_class, dtype=torch.long)


class StabilityPredictor(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.3):
        super(StabilityPredictor, self).__init__()

        # Default pre-trained weights
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.efficientnet = models.efficientnet_v2_s(weights=weights)

        # Get the number of input features to the final classifier layer
        num_ftrs = self.efficientnet.classifier[1].in_features

        # Replace the default classifier with a custom one (Dropout + Linear layer)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)


class EfficientAttentionNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.3):
        super(EfficientAttentionNet, self).__init__()

        # Default pre-trained weights for EfficientNet V2 Small
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.efficientnet = models.efficientnet_v2_s(weights=weights)

        # Spatial attention module
        self.spatial_attention = SpatialAttentionModule(kernel_size=7)

        # Get the number of input features to the final classifier layer
        num_ftrs = self.efficientnet.classifier[1].in_features

        # Replace the default classifier with a custom one (Dropout + Linear layer)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        # Pass through the feature extractor (EfficientNet backbone) until the last feature map
        features = self.efficientnet.features(x)  # Extract convolutional features
        
        # Apply spatial attention module to the feature maps
        features = self.spatial_attention(features)
        
        # Global average pooling (same as EfficientNet)
        pooled_features = self.efficientnet.avgpool(features)
        
        # Flatten the pooled features
        pooled_features = torch.flatten(pooled_features, 1)
        
        # Pass through the custom classifier
        output = self.efficientnet.classifier(pooled_features)
        
        return output



class ConvnextPredictor(nn.Module):
    def __init__(self, num_classes=6, freeze_layers=True):
        super(ConvnextPredictor, self).__init__()

        # Default pre-trained weights
        weights = models.convnext.ConvNeXt_Base_Weights.DEFAULT
        self.convnextnet = convnext_base(weights=weights)

        # Force classifier to have only 6 outputs
        num_ftrs = self.convnextnet.classifier[2].in_features
        self.convnextnet.classifier[2] = nn.Linear(num_ftrs, out_features=num_classes, bias=True)

        if freeze_layers:
            # Freeze ConvNeXt backbone layers for quicker fine-tuning training
            for param in self.convnextnet.parameters():
                param.requires_grad = False

            # Only unfreeze the classifier layers
            for param in self.convnextnet.classifier.parameters():
                param.requires_grad = True


    def forward(self, x):
        return self.convnextnet(x)
    

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise max and average pooling (along spatial dimensions)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(concat))
        return x * attention_map


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, device):
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

def calculate_stats(dataset):
    loader = DataLoader(dataset, batch_size=100, num_workers=0, shuffle=False)
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    
    mean /= len(dataset)
    std /= len(dataset)
    return mean, std

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

def train_and_save(config):
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create full dataset without normalization and augmentation
    full_dataset = StabilityDataset(csv_file=config['train_csv'], 
                                    img_dir=config['train_img_dir'], 
                                    transform=transforms.ToTensor(),
                                    augment=False,
                                    image_size=config['image_size'])

    # Split dataset into train and validation
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(config['val_ratio'] * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    # Calculate statistics for training and validation sets separately
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    print("Calculating training dataset statistics...")
    train_mean, train_std = calculate_stats(train_subset)
    print(f"Training dataset mean: {train_mean}")
    print(f"Training dataset std: {train_std}")

    print("Calculating validation dataset statistics...")
    val_mean, val_std = calculate_stats(val_subset)
    print(f"Validation dataset mean: {val_mean}")
    print(f"Validation dataset std: {val_std}")

    # Create separate transforms for training and validation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=val_mean, std=val_std),
    ])

    # Create augmented training dataset and non-augmented validation dataset
    train_dataset = StabilityDataset(csv_file=config['train_csv'], 
                                     img_dir=config['train_img_dir'], 
                                     transform=train_transform,
                                     augment=config['use_augmentation'],
                                     image_size=config['image_size'])
    train_dataset = Subset(train_dataset, [i for i in range(len(train_dataset)) if i % len(full_dataset) in train_indices])

    val_dataset = StabilityDataset(csv_file=config['train_csv'], 
                                   img_dir=config['train_img_dir'], 
                                   transform=val_transform,
                                   augment=False,
                                   image_size=config['image_size'])
    val_dataset = Subset(val_dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # Initialize model, criterion, optimizer, and scheduler
    model = StabilityPredictor(num_classes=config['num_classes'], dropout_rate=config['dropout_rate'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_factor'], patience=config['lr_patience'], verbose=True)

    # Train model
    print('Training...')
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                        num_epochs=config['num_epochs'], patience=config['early_stopping_patience'], device=device)

    torch.save(model.state_dict(), config['model_save_path'])
    print("Training complete. Model saved.")

    # Prediction on test set
    test_dataset = StabilityDataset(csv_file=config['test_csv'],
                                    img_dir=config['test_img_dir'],
                                    transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    predictions, image_ids = predict(model, test_loader, device)

    # Save predictions to CSV
    with open(config['predictions_save_path'], 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'labels'])
        for img_id, pred in zip(image_ids, predictions):
            writer.writerow([int(img_id) + 1, int(pred)])  # Ensure both are integers
    print(f"Predictions saved to {config['predictions_save_path']}")

# Windows can't do multicore processing
def get_optimal_num_workers():
    if platform.system() == 'Windows':
        return 0
    else:
        return multiprocessing.cpu_count()

# Hyperparameters
config = {
    'train_csv': './COMP90086_2024_Project_train/train.csv',
    'train_img_dir': './COMP90086_2024_Project_train/train',
    'test_csv': './COMP90086_2024_Project_test/test.csv',
    'test_img_dir': './COMP90086_2024_Project_test/test',
    'image_size': 224,
    'val_ratio': 0.05,
    'use_augmentation': True,
    'batch_size': 32,
    'num_workers': get_optimal_num_workers(),
    'num_classes': 6,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'lr_factor': 0.1,
    'lr_patience': 2,
    'num_epochs': 30,
    'early_stopping_patience': 5,
    'model_save_path': 'stability_predictor_efficientnetv2.pth',
    'predictions_save_path': 'predictions.csv'
}