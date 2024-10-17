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
import cv2
import json


class StabilityDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, augment=False, use_quantized=False):
        self.stability_data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        self.use_quantized = use_quantized
        self.image_files = self._get_image_files()

    def _get_image_files(self):
        image_files = []
        for idx, row in self.stability_data.iterrows():
            img_name = str(row[0])
            if self.use_quantized:
                image_files.append(f"quantized/{img_name}_quantized.jpg")
                if self.augment:
                    image_files.extend([
                        f"quantized/{img_name}_flipped_quantized.jpg",
                        f"quantized/{img_name}_zoomed_quantized.jpg",
                        f"quantized/{img_name}_zoomed_flipped_quantized.jpg"
                    ])
            else:
                image_files.append(f"{img_name}_original.jpg")
                if self.augment:
                    image_files.extend([
                        f"{img_name}_flipped.jpg",
                        f"{img_name}_zoomed.jpg",
                        f"{img_name}_zoomed_flipped.jpg"
                    ])
        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_idx = idx // 4 if self.augment else idx
        stability_height = self.stability_data.iloc[original_idx, -1]
        stability_class = int(stability_height) - 1

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

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



class ChannelAttentionModule(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class EfficientChannelAttentionNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.0):
        super(EfficientChannelAttentionNet, self).__init__()

        # Default pre-trained weights for EfficientNet V2 Small
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.efficientnet = models.efficientnet_v2_s(weights=weights)

        # Add channel attention modules after specific layers in the EfficientNet backbone
        # Adjust `in_planes` to match the number of channels at each stage of EfficientNet
        self.channel_attention1 = ChannelAttentionModule(in_planes=24)  # After first block (features[1])
        self.channel_attention2 = ChannelAttentionModule(in_planes=48)  # After second block (features[2])

        # Get the number of input features to the final classifier layer
        num_ftrs = self.efficientnet.classifier[1].in_features

        # Replace the default classifier with a custom one (Dropout + Linear layer)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        # Pass input through the first few layers of EfficientNet
        x = self.efficientnet.features[0](x)  # Initial convolution and stem
        x = self.efficientnet.features[1](x)  # First block (channels: 24)
        x = self.channel_attention1(x)  # Apply channel attention after the first block
        
        x = self.efficientnet.features[2](x)  # Second block (channels: 48)
        x = self.channel_attention2(x)  # Apply channel attention after the second block
        
        # Continue with the rest of the EfficientNet layers
        for i in range(3, len(self.efficientnet.features)):
            x = self.efficientnet.features[i](x)

        # Global average pooling and final classifier
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.efficientnet.classifier(x)

        return x



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
            print('layers frozen!')
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



def colour_quantisation(image, k=20):
    # Convert the image to 2D pixel array
    pixels = np.float32(image.reshape(-1, 3))

    # Define criteria for K-Means (stop after 10 iter or if accuracy reaches 1.0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Apply K-Means clustering
    _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    quantised = np.uint8(palette)[labels.flatten()]

    # Reshape the image to original dimensions
    quantised = quantised.reshape(image.shape)
    
    return quantised


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

def save_split_and_stats(train_indices, val_indices, train_mean, train_std, val_mean, val_std, filename):
    data = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'train_mean': train_mean.tolist(),
        'train_std': train_std.tolist(),
        'val_mean': val_mean.tolist(),
        'val_std': val_std.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_split_and_stats(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return (
        data['train_indices'],
        data['val_indices'],
        torch.tensor(data['train_mean']),
        torch.tensor(data['train_std']),
        torch.tensor(data['val_mean']),
        torch.tensor(data['val_std'])
    )

def train_and_save(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check if we should load existing split and stats
    split_stats_file = config.get('split_stats_file', 'split_and_stats.json')
    if config.get('use_existing_split', False) and os.path.exists(split_stats_file):
        print(f"Loading existing split and stats from {split_stats_file}")
        train_indices, val_indices, train_mean, train_std, val_mean, val_std = load_split_and_stats(split_stats_file)
    else:
        # Create a dataset with only original images for statistics calculation
        stats_dataset = StabilityDataset(csv_file=config['train_csv'], 
                                         img_dir=config['train_img_dir'], 
                                         transform=transforms.ToTensor(),
                                         augment=False,
                                         use_quantized=config['use_quantized'])

        # Split dataset into train and validation
        dataset_size = len(stats_dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        split = int(np.floor(config['val_ratio'] * dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]

        # Calculate statistics for training and validation sets
        train_subset = Subset(stats_dataset, train_indices)
        val_subset = Subset(stats_dataset, val_indices)

        print("Calculating training dataset statistics...")
        train_mean, train_std = calculate_stats(train_subset)
        print(f"Training dataset mean: {train_mean}")
        print(f"Training dataset std: {train_std}")

        print("Calculating validation dataset statistics...")
        val_mean, val_std = calculate_stats(val_subset)
        print(f"Validation dataset mean: {val_mean}")
        print(f"Validation dataset std: {val_std}")

        # Save split and stats
        save_split_and_stats(train_indices, val_indices, train_mean, train_std, val_mean, val_std, split_stats_file)
        print(f"Split and stats saved to {split_stats_file}")

    # Create transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=val_mean, std=val_std),
    ])

    # Create full datasets with augmentation
    train_dataset = StabilityDataset(csv_file=config['train_csv'], 
                                     img_dir=config['train_img_dir'], 
                                     transform=train_transform,
                                     augment=config['use_augmentation'],
                                     use_quantized=config['use_quantized'])

    val_dataset = StabilityDataset(csv_file=config['train_csv'], 
                                   img_dir=config['train_img_dir'], 
                                   transform=val_transform,
                                   augment=False,
                                   use_quantized=config['use_quantized'])

    # Apply the split
    train_dataset = Subset(train_dataset, [i for i in range(len(train_dataset)) if i % len(val_dataset) in train_indices])
    val_dataset = Subset(val_dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # Initialize model, criterion, optimizer, and scheduler
    if config['model'] == 'StabilityPredictor':
        model = StabilityPredictor(num_classes=config['num_classes'], dropout_rate=config['dropout_rate'])
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['model'] == 'EfficientAttentionNet':
        model = EfficientAttentionNet(dropout_rate=config['dropout_rate'])
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['model'] == 'EfficientChannelAttentionNet':
        model = EfficientChannelAttentionNet(dropout_rate=config['dropout_rate'])
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['model'] == 'ConvnextPredictor':
        model = ConvnextPredictor(num_classes=config['num_classes'], freeze_layers=config['freeze_layers'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    else:
        print('Unrecognised model in config. Defaulting to StabilityPredictor (EfficientNet)')
        model = StabilityPredictor(num_classes=config['num_classes'], dropout_rate=config['dropout_rate'])
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    print('Model: ' + config['model'])
    print('Using quantized images: ' + str(config['use_quantized']))

    criterion = nn.CrossEntropyLoss()
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
                                    transform=val_transform,
                                    use_quantized=config['use_quantized'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    predictions, image_ids = predict(model, test_loader, device)

    # Save predictions to CSV
    with open(config['predictions_save_path'], 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'labels'])
        for img_id, pred in zip(image_ids, predictions):
            writer.writerow([int(img_id) + 1, int(pred)])
    print(f"Predictions saved to {config['predictions_save_path']}")

# Windows can't do multicore processing
def get_optimal_num_workers():
    if platform.system() == 'Windows':
        return 0
    else:
        return multiprocessing.cpu_count() - 2