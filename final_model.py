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
import uuid

class StabilityDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, augment=False, use_quantized=False, additional_columns=None, target_column=None, balance_dataset=False, reference_csv=None):
        self.stability_data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        self.use_quantized = use_quantized
        self.additional_columns = additional_columns or []
        self.target_column = target_column
        self.image_files = self._get_image_files()
        self.feature_categories = self._get_feature_categories()

        # Use reference_csv for label mapping if provided (for test set)
        if reference_csv and target_column:
            reference_data = pd.read_csv(reference_csv)
            self.unique_labels = sorted(reference_data[target_column].unique())
        elif target_column and target_column in self.stability_data.columns:
            self.unique_labels = sorted(self.stability_data[target_column].unique())
        else:
            self.unique_labels = None

        if self.unique_labels:
            self.label_to_index = {label: index for index, label in enumerate(self.unique_labels)}
            self.index_to_label = {index: label for label, index in self.label_to_index.items()}
        else:
            self.label_to_index = None
            self.index_to_label = None

        if balance_dataset and self.target_column is not None:
            self._balance_dataset()

    def _balance_dataset(self):
        # Count occurrences of each class
        class_counts = self.stability_data[self.target_column].value_counts()
        min_class_count = class_counts.min()

        # Undersample each class
        balanced_data = []
        for class_label in class_counts.index:
            class_data = self.stability_data[self.stability_data[self.target_column] == class_label]
            balanced_data.append(class_data.sample(min_class_count, replace=False))

        # Combine the balanced classes
        self.stability_data = pd.concat(balanced_data).reset_index(drop=True)

        # Update image files based on the balanced dataset
        self.image_files = self._get_image_files()


    def _get_image_files(self):
        image_files = []
        for idx, row in self.stability_data.iterrows():
            img_name = str(row.iloc[0])
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

    def _get_feature_categories(self):
        feature_categories = {}
        for col in self.additional_columns:
            if col in self.stability_data.columns:
                unique_values = self.stability_data[col].unique()
                feature_categories[col] = {
                    'num_categories': len(unique_values),
                    'value_to_index': {val: idx for idx, val in enumerate(unique_values)}
                }
        if self.target_column and self.target_column in self.stability_data.columns:
            unique_values = self.stability_data[self.target_column].unique()
            feature_categories[self.target_column] = {
                'num_categories': len(unique_values),
                'value_to_index': {val: idx for idx, val in enumerate(unique_values)}
            }
        return feature_categories

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_idx = idx // 4 if self.augment else idx
        image_id = self.stability_data.iloc[original_idx, 0]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

        additional_data = []
        for col in self.additional_columns:
            if col in self.stability_data.columns:
                value = self.stability_data.iloc[original_idx][col]
                index = self.feature_categories[col]['value_to_index'][value]
                additional_data.append(torch.tensor(index, dtype=torch.long))

        if self.target_column and self.target_column in self.stability_data.columns:
            target_value = self.stability_data.iloc[original_idx][self.target_column]
            target_index = self.label_to_index[target_value]
            return (image, image_id, torch.tensor(target_index, dtype=torch.long), *additional_data)
        else:
            return (image, image_id, *additional_data)


    def get_feature_dimensions(self):
        return {col: info['num_categories'] for col, info in self.feature_categories.items() if col != self.target_column}

    def get_target_dimension(self):
        if self.target_column and self.target_column in self.stability_data.columns:
            return len(self.unique_labels)
        return None

    def get_original_label(self, index):
        if self.index_to_label is None:
            raise ValueError("Label mapping is not available")
        return self.index_to_label[index]

class StabilityPredictor(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3, additional_features=None):
        super(StabilityPredictor, self).__init__()

        # Default pre-trained weights
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.efficientnet = models.efficientnet_v2_s(weights=weights)

        # Get the number of input features to the final classifier layer
        num_ftrs = self.efficientnet.classifier[1].in_features

        # Embedding layers for additional features
        self.additional_features = additional_features or {}
        self.embedding_layers = nn.ModuleDict()
        self.embedding_dim = 16  # You can adjust this value
        total_embedding_dim = 0

        for feature, num_categories in self.additional_features.items():
            self.embedding_layers[feature] = nn.Embedding(num_categories, self.embedding_dim)
            total_embedding_dim += self.embedding_dim

        # Combine image features with embeddings
        self.combined_layer = nn.Linear(num_ftrs + total_embedding_dim, num_ftrs)

        # Replace the default classifier with a custom one (Dropout + Linear layer)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x, *additional_inputs):
        # Process the image through EfficientNet
        x = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)

        # Process additional features through embedding layers
        embeddings = []
        for i, (feature, _) in enumerate(self.additional_features.items()):
            embedding = self.embedding_layers[feature](additional_inputs[i])
            embeddings.append(embedding)

        # Concatenate image features with embeddings
        if embeddings:
            x = torch.cat([x] + embeddings, dim=1)
            x = self.combined_layer(x)

        # Final classification
        x = self.classifier(x)
        return x


class EfficientAttentionNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3, additional_features=None):
        super(EfficientAttentionNet, self).__init__()

        # Default pre-trained weights for EfficientNet V2 Small
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.efficientnet = models.efficientnet_v2_s(weights=weights)

        # Spatial attention module
        self.spatial_attention = SpatialAttentionModule(kernel_size=7)

        # Get the number of input features to the final classifier layer
        num_ftrs = self.efficientnet.classifier[1].in_features

        # Embedding layers for additional features
        self.additional_features = additional_features or {}
        self.embedding_layers = nn.ModuleDict()
        self.embedding_dim = 16
        total_embedding_dim = 0

        for feature, num_categories in self.additional_features.items():
            self.embedding_layers[feature] = nn.Embedding(num_categories, self.embedding_dim)
            total_embedding_dim += self.embedding_dim

        # Combine image features with embeddings
        self.combined_layer = nn.Linear(num_ftrs + total_embedding_dim, num_ftrs)

        # Replace the default classifier with a custom one (Dropout + Linear layer)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x, *additional_inputs):
        # Pass through the feature extractor (EfficientNet backbone) until the last feature map
        features = self.efficientnet.features(x)  # Extract convolutional features
        
        # Apply spatial attention module to the feature maps
        features = self.spatial_attention(features)
        
        # Global average pooling
        x = self.efficientnet.avgpool(features)
        
        # Flatten the pooled features
        x = torch.flatten(x, 1)

        # Process additional features through embedding layers
        embeddings = []
        for i, (feature, _) in enumerate(self.additional_features.items()):
            embedding = self.embedding_layers[feature](additional_inputs[i])
            embeddings.append(embedding)

        # Concatenate image features with embeddings
        if embeddings:
            x = torch.cat([x] + embeddings, dim=1)
            x = self.combined_layer(x)

        # Final classification
        x = self.classifier(x)
        return x

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

class EfficientChannelAttentionNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.0, additional_features=None):
        super(EfficientChannelAttentionNet, self).__init__()

        # Default pre-trained weights for EfficientNet V2 Small
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.efficientnet = models.efficientnet_v2_s(weights=weights)

        # Add channel attention modules after specific layers in the EfficientNet backbone
        self.channel_attention1 = ChannelAttentionModule(in_planes=24)  # After first block (features[1])
        self.channel_attention2 = ChannelAttentionModule(in_planes=48)  # After second block (features[2])

        # Get the number of input features to the final classifier layer
        num_ftrs = self.efficientnet.classifier[1].in_features

        # Embedding layers for additional features
        self.additional_features = additional_features or {}
        self.embedding_layers = nn.ModuleDict()
        self.embedding_dim = 16
        total_embedding_dim = 0

        for feature, num_categories in self.additional_features.items():
            self.embedding_layers[feature] = nn.Embedding(num_categories, self.embedding_dim)
            total_embedding_dim += self.embedding_dim

        # Combine image features with embeddings
        self.combined_layer = nn.Linear(num_ftrs + total_embedding_dim, num_ftrs)

        # Replace the default classifier with a custom one (Dropout + Linear layer)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x, *additional_inputs):
        # Pass input through the first few layers of EfficientNet
        x = self.efficientnet.features[0](x)  # Initial convolution and stem
        x = self.efficientnet.features[1](x)  # First block (channels: 24)
        x = self.channel_attention1(x)  # Apply channel attention after the first block
        
        x = self.efficientnet.features[2](x)  # Second block (channels: 48)
        x = self.channel_attention2(x)  # Apply channel attention after the second block
        
        # Continue with the rest of the EfficientNet layers
        for i in range(3, len(self.efficientnet.features)):
            x = self.efficientnet.features[i](x)

        # Global average pooling
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)

        # Process additional features through embedding layers
        embeddings = []
        for i, (feature, _) in enumerate(self.additional_features.items()):
            embedding = self.embedding_layers[feature](additional_inputs[i])
            embeddings.append(embedding)

        # Concatenate image features with embeddings
        if embeddings:
            x = torch.cat([x] + embeddings, dim=1)
            x = self.combined_layer(x)

        # Final classification
        x = self.classifier(x)

        return x

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
    

class EfficientSpatialChannelAttentionNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.01, additional_features = None):
        super(EfficientSpatialChannelAttentionNet, self).__init__()

        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.efficientnet = models.efficientnet_v2_s(weights=weights)

        self.channel_attention1 = ChannelAttentionModule(in_planes=24)  # inputs for efficientnet block 1
        self.spatial_attention1 = SpatialAttentionModule(kernel_size=7)
        
        self.channel_attention2 = ChannelAttentionModule(in_planes=48)  # inputs for efficientnet block 2
        self.spatial_attention2 = SpatialAttentionModule(kernel_size=7)

        # Get num of inputs for final classifier layer
        num_ftrs = self.efficientnet.classifier[1].in_features


        # Embedding layers for additional features
        self.additional_features = additional_features or {}
        self.embedding_layers = nn.ModuleDict()
        self.embedding_dim = 16
        total_embedding_dim = 0

        for feature, num_categories in self.additional_features.items():
            self.embedding_layers[feature] = nn.Embedding(num_categories, self.embedding_dim)
            total_embedding_dim += self.embedding_dim

        # Combine image features with embeddings
        self.combined_layer = nn.Linear(num_ftrs + total_embedding_dim, num_ftrs)

        # Replace the default classifier with a custom one (Dropout + Linear layer)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x, *additional_inputs):
        # Initial convolution and stem
        x = self.efficientnet.features[0](x)

        x = self.efficientnet.features[1](x)
        x = self.channel_attention1(x)
        x = self.spatial_attention1(x)

        x = self.efficientnet.features[2](x)
        x = self.channel_attention2(x)
        x = self.spatial_attention2(x)

        # keep forward passing as normal after attention
        for i in range(3, len(self.efficientnet.features)):
            x = self.efficientnet.features[i](x)

        # Global average pooling and final classifier
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)

        # Process additional features through embedding layers
        embeddings = []
        for i, (feature, _) in enumerate(self.additional_features.items()):
            embedding = self.embedding_layers[feature](additional_inputs[i])
            embeddings.append(embedding)

        # Concatenate image features with embeddings
        if embeddings:
            x = torch.cat([x] + embeddings, dim=1)
            x = self.combined_layer(x)

        # Final classification
        x = self.classifier(x)

        return x


class ConvnextPredictor(nn.Module):
    def __init__(self, num_classes=6, freeze_layers=True, additional_features=None):
        super(ConvnextPredictor, self).__init__()

        # Default pre-trained weights
        weights = models.convnext.ConvNeXt_Base_Weights.DEFAULT
        self.convnextnet = convnext_base(weights=weights)

        # Get the number of input features to the final classifier layer
        num_ftrs = self.convnextnet.classifier[2].in_features

        # Embedding layers for additional features
        self.additional_features = additional_features or {}
        self.embedding_layers = nn.ModuleDict()
        self.embedding_dim = 16  # You can adjust this value
        total_embedding_dim = 0

        for feature, num_categories in self.additional_features.items():
            self.embedding_layers[feature] = nn.Embedding(num_categories, self.embedding_dim)
            total_embedding_dim += self.embedding_dim

        # Combine ConvNeXt features with embeddings
        self.combined_layer = nn.Linear(num_ftrs + total_embedding_dim, num_ftrs)

        # Replace the default classifier with a custom one
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_ftrs),  # ConvNeXt uses LayerNorm instead of BatchNorm
            nn.Flatten(start_dim=1),
            nn.Linear(num_ftrs, num_classes)
        )

        if freeze_layers:
            print('Layers frozen!')
            # Freeze ConvNeXt backbone layers for quicker fine-tuning training
            for param in self.convnextnet.parameters():
                param.requires_grad = False

            # Only unfreeze the classifier layers and the combined layer
            for param in self.classifier.parameters():
                param.requires_grad = True
            for param in self.combined_layer.parameters():
                param.requires_grad = True
            for embedding_layer in self.embedding_layers.values():
                for param in embedding_layer.parameters():
                    param.requires_grad = True

    def forward(self, x, *additional_inputs):
        # Pass through ConvNeXt backbone
        x = self.convnextnet.features(x)
        x = self.convnextnet.avgpool(x)
        x = torch.flatten(x, 1)

        # Process additional features through embedding layers
        embeddings = []
        for i, (feature, _) in enumerate(self.additional_features.items()):
            embedding = self.embedding_layers[feature](additional_inputs[i])
            embeddings.append(embedding)

        # Concatenate ConvNeXt features with embeddings
        if embeddings:
            x = torch.cat([x] + embeddings, dim=1)
            x = self.combined_layer(x)

        # Final classification
        x = self.classifier(x)

        return x

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

def train_model_full_dataset(model, train_loader, criterion, optimizer, scheduler, num_epochs, device, lr_schedule):
    model.to(device)
    
    # Get the learning rate for each epoch
    def get_lr(epoch):
        return lr_schedule.get(epoch, lr_schedule[max(k for k in lr_schedule.keys() if k <= epoch)])

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Set the learning rate for this epoch
        current_lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Training phase
        model.train()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_training=True)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        print('-' * 60)

    return model

def train_model_split(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, device):
    model.to(device)
    id = generate_model_id()
    epochs_no_improve = 0
    best_epoch = None
    lr_schedule = {0: optimizer.param_groups[0]['lr']}  # Initial learning rate
    
    best_val_loss = float('inf')
    best_val_acc = 0
    best_val_class_performance = None
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_training=True)
        
        # Validation phase
        model.eval()
        val_loss, val_acc, val_class_performance = run_epoch(model, val_loader, criterion, optimizer, device, is_training=False, return_class_performance=True)
        
        # Learning rate scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # If learning rate changed, add to schedule
        if new_lr != old_lr:
            lr_schedule[epoch + 1] = new_lr

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {scheduler.optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 60)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_class_performance = val_class_performance
            best_epoch = epoch
            epochs_no_improve = 0
            
            torch.save(model.state_dict(), f'state_dicts/{id}.pth')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    return id, model, best_epoch, lr_schedule, best_val_loss, best_val_acc, best_val_class_performance

def run_epoch(model, data_loader, criterion, optimizer, device, is_training=True, return_class_performance=False):
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = torch.zeros(model.classifier[-1].out_features, device=device)
    class_total = torch.zeros(model.classifier[-1].out_features, device=device)

    # Create progress bar
    progress_bar = tqdm(data_loader, desc="Training" if is_training else "Validating")

    for batch in progress_bar:
        inputs = batch[0].to(device)
        labels = batch[2].to(device)
        additional_inputs = [feature.to(device) for feature in batch[3:]]  # Change this from batch[2:] to batch[3:]
        
        if is_training:
            optimizer.zero_grad()
        
        outputs = model(inputs, *additional_inputs)
        loss = criterion(outputs, labels)
        
        if is_training:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Per-class performance
        for label, pred in zip(labels, predicted):
            class_total[label] += 1
            class_correct[label] += (label == pred).item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = 100. * correct / total

    if return_class_performance:
        class_performance = (100. * class_correct / class_total).cpu().numpy()
        return epoch_loss, epoch_acc, class_performance
    else:
        return epoch_loss, epoch_acc

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    image_ids = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            ids = batch[1]
            additional_inputs = [feature.to(device) for feature in batch[2:]]
            outputs = model(inputs, *additional_inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            image_ids.extend(ids.numpy())
    
    return predictions, image_ids

def calculate_stats(dataset):
    loader = DataLoader(dataset, batch_size=100, num_workers=get_optimal_num_workers(), shuffle=False)
    mean = 0.
    std = 0.
    total_samples = len(dataset)
    
    # Create a tqdm progress bar
    pbar = tqdm(total=total_samples, desc="Calculating Stats", unit="sample")
    
    for batch in loader:
        images = batch[0]  # Assuming images are always the first element
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        
        # Update the progress bar
        pbar.update(batch_samples)
   
    mean /= total_samples
    std /= total_samples
    
    # Close the progress bar
    pbar.close()
    
    return mean, std

def check_label_consistency(train_csv, predictions_csv, target_column):
    train_data = pd.read_csv(train_csv)
    pred_data = pd.read_csv(predictions_csv)
    
    train_labels = set(train_data[target_column].unique())
    pred_labels = set(pred_data[target_column].unique())
    
    if train_labels == pred_labels:
        print("Labels are consistent between training set and predictions.")
    else:
        print("Warning: Label mismatch detected!")
        print("Training labels:", train_labels)
        print("Prediction labels:", pred_labels)
        print("Difference:", train_labels.symmetric_difference(pred_labels))


def save_split_and_stats(train_indices, val_indices, train_mean, train_std, filename):
    data = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'train_mean': train_mean.tolist(),
        'train_std': train_std.tolist()
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
        torch.tensor(data['train_std'])
    )

# Windows can't do multicore processing
def get_optimal_num_workers():
    if platform.system() == 'Windows':
        return 0
    else:
        return multiprocessing.cpu_count()

def load_or_create_split_and_stats(config, use_full_dataset):
    full_data_suffix = "_full" if use_full_dataset else ""
    split_stats_file = f"{config['output_folder']}/split_and_stats{full_data_suffix}.json"
    
    if config['use_existing_split'] and os.path.exists(split_stats_file):
        print(f"Loading existing split and stats from {split_stats_file}")
        return load_split_and_stats(split_stats_file)
    else:
        return create_new_split_and_stats(config, use_full_dataset, split_stats_file)

def create_new_split_and_stats(config, use_full_dataset, split_stats_file):
    base_dataset = create_base_dataset(config)
    train_mean, train_std = calculate_stats(base_dataset)
    print(f"Dataset mean: {train_mean}")
    print(f"Dataset std: {train_std}")

    if use_full_dataset:
        train_indices = list(range(len(base_dataset)))
        val_indices = []
    else:
        train_indices, val_indices = split_dataset(base_dataset, config['val_ratio'])

    save_split_and_stats(train_indices, val_indices, train_mean, train_std, split_stats_file)
    print(f"Split and stats saved to {split_stats_file}")
    return train_indices, val_indices, train_mean, train_std

def create_base_dataset(config):
    return StabilityDataset(
        csv_file=config['train_csv'], 
        img_dir=config['train_img_dir'], 
        transform=transforms.ToTensor(),
        augment=False,
        use_quantized=config['use_quantized'],
        target_column=config['target_column'],
        additional_columns=config['additional_columns'],
        balance_dataset=config['balance_dataset']
    )

def split_dataset(dataset, val_ratio):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(val_ratio * dataset_size))
    return indices[split:], indices[:split]

def create_datasets_and_loaders(config, train_indices, val_indices, train_mean, train_std, use_full_dataset):
    base_transform = create_transforms(train_mean, train_std)
    full_dataset = create_full_dataset(config, base_transform)
    
    train_dataset, val_dataset = create_train_val_datasets(full_dataset, train_indices, val_indices, config['use_augmentation'], use_full_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=get_optimal_num_workers())
    val_loader = None if use_full_dataset else DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=get_optimal_num_workers())
    
    return full_dataset, train_loader, val_loader

def create_transforms(train_mean, train_std):
    normalize_transform = transforms.Normalize(mean=train_mean, std=train_std)
    return transforms.Compose([transforms.ToTensor(), normalize_transform])

def create_full_dataset(config, base_transform):
    return StabilityDataset(
        csv_file=config['train_csv'], 
        img_dir=config['train_img_dir'], 
        transform=base_transform,
        augment=config['use_augmentation'],
        use_quantized=config['use_quantized'],
        additional_columns=config['additional_columns'],
        target_column=config['target_column'],
        balance_dataset=config['balance_dataset']
    )

def create_train_val_datasets(full_dataset, train_indices, val_indices, use_augmentation, use_full_dataset):
    if use_augmentation:
        train_indices = [i for idx in train_indices for i in range(idx * 4, (idx + 1) * 4)]
        val_indices = [i for idx in val_indices for i in range(idx * 4, (idx + 1) * 4)]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = None if use_full_dataset else Subset(full_dataset, val_indices[:len(val_indices)//4])
    
    return train_dataset, val_dataset

def initialize_model(config, num_classes, additional_features):
    model_class = get_model_class(config['model'])
    return model_class(num_classes=num_classes, dropout_rate=config['dropout_rate'], additional_features=additional_features)

def get_model_class(model_name):
    model_classes = {
        'StabilityPredictor': StabilityPredictor,
        'EfficientAttentionNet': EfficientAttentionNet,
        'EfficientChannelAttentionNet': EfficientChannelAttentionNet,
        'ConvnextPredictor': ConvnextPredictor,
        'EfficientSpatialChannelAttentionNet': EfficientSpatialChannelAttentionNet
    }
    return model_classes.get(model_name, StabilityPredictor)

def train_full_dataset(id, model, config, train_loader, criterion, device, training_params):
    num_epochs = training_params['epochs']
    lr_schedule = training_params['lr_schedule']
    
    # Ensure lr_schedule keys are integers
    lr_schedule = {int(k): float(v) for k, v in lr_schedule.items()}
    
    initial_lr = lr_schedule[0]  # Assume the first key is 0 for initial learning rate
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=config['weight_decay'])
    
    model = train_model_full_dataset(model, train_loader, criterion, optimizer, None, num_epochs, device, lr_schedule)
    torch.save(model.state_dict(), f'state_dicts/full_{id}.pth')
    return id, model

def train_with_validation(model, config, train_loader, val_loader, criterion, device):
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_factor'], patience=config['lr_patience'])
    
    id, model, best_epoch, lr_schedule, best_val_loss, best_val_acc, best_val_class_performance = train_model_split(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        num_epochs=config['num_epochs'], patience=config['early_stopping_patience'], device=device
    )
    
    save_training_results(id, config, best_epoch, lr_schedule, best_val_loss, best_val_acc, best_val_class_performance)
    
    return id, model

def run_predictions(model, config, base_transform, device, use_full_dataset):
    test_dataset = create_test_dataset(config, base_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=get_optimal_num_workers())
    
    predictions, image_ids = predict(model, test_loader, device)
    save_predictions_to_csv(config, image_ids, predictions, use_full_dataset, test_dataset)

def create_test_dataset(config, base_transform):
    return StabilityDataset(
        csv_file=config['test_csv'],
        img_dir=config['test_img_dir'],
        transform=base_transform,
        augment=False,
        use_quantized=config['use_quantized'],
        additional_columns=config['additional_columns'],
        target_column=config['target_column'],
        reference_csv=config['train_csv']
    )

def train_and_save(config, use_full_dataset=False, do_predictions=False, model_id=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_indices, val_indices, train_mean, train_std = load_or_create_split_and_stats(config, use_full_dataset)
    full_dataset, train_loader, val_loader = create_datasets_and_loaders(config, train_indices, val_indices, train_mean, train_std, use_full_dataset)

    additional_features = full_dataset.get_feature_dimensions()
    num_classes = full_dataset.get_target_dimension()

    if use_full_dataset:
        # Load config and training params from model_results.csv
        results_file = f"{config['output_folder']}/model_results.csv"
        config, training_params = load_config_from_results(config, results_file, model_id)

    model = initialize_model(config, num_classes, additional_features)
    print_model_info(config, additional_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    
    if use_full_dataset:
        id, model = train_full_dataset(model_id, model, config, train_loader, criterion, device, training_params)
    else:
        id, model = train_with_validation(model, config, train_loader, val_loader, criterion, device)

    if do_predictions:
        base_transform = create_transforms(train_mean, train_std)
        run_predictions(model, config, base_transform, device, use_full_dataset)

    return id

def print_model_info(config, additional_features, num_classes):
    print('Model:', config['model'])
    if len(additional_features) > 0:
        print('Additional features:', ', '.join(f"{k}: {v} categories" for k, v in additional_features.items()))
    print(f'Target feature: {config["target_column"]} ({num_classes} categories)')

def save_predictions_to_csv(config, image_ids, predictions, use_full_dataset, test_dataset):
    model_type = "full" if use_full_dataset else "val"
    predictions_save_path = f"{config['output_folder']}/{model_type}_predictions.csv"
    
    # Check if file exists and if target column is present
    file_exists = os.path.isfile(predictions_save_path)
    target_column_exists = False
    
    if file_exists:
        df = pd.read_csv(predictions_save_path)
        target_column_exists = config['target_column'] in df.columns
    
    # Prepare new data
    new_data = pd.DataFrame({
        'id': image_ids,
        config['target_column']: [int(test_dataset.get_original_label(pred)) for pred in predictions]
    })
    
    if file_exists and target_column_exists:
        # Update existing file
        df = pd.read_csv(predictions_save_path)
        df = df.set_index('id')
        df.update(new_data.set_index('id'))
        df.to_csv(predictions_save_path)
    else:
        # Create new file or append column
        if file_exists:
            df = pd.read_csv(predictions_save_path)
            df = df.merge(new_data, on='id', how='left')
        else:
            df = new_data
        df.to_csv(predictions_save_path, index=False)
    
    print(f"Predictions saved to {predictions_save_path}")

def save_training_results(id, config, best_epoch, lr_schedule, best_val_loss, best_val_acc, best_val_class_performance):
    training_params = {
        "epochs": best_epoch + 1,
        "initial_lr": config['learning_rate'],
        "best_val_loss": best_val_loss,
        "lr_schedule": {str(k): v for k, v in lr_schedule.items()},
    }
    record_results(id, config, best_val_loss, best_val_acc, best_val_class_performance, training_params, f"{config['output_folder']}/model_results.csv")

def generate_model_id():
    return str(uuid.uuid4())

def record_results(id, config, best_val_loss, best_val_acc, best_val_class_performance, training_params, file_path):
    fieldnames = [
        'id', 'model', 'target_column', 'additional_columns', 'balance_dataset', 'use_augmentation', 
        'use_quantized', 'val_ratio', 'batch_size', 'dropout_rate', 'learning_rate', 'lr_factor', 
        'lr_patience', 'freeze_layers', 'num_epochs', 'use_existing_split', 'early_stopping_patience', 
        'weight_decay', 'val_loss', 'val_acc'
    ] + [f'class_{i}_acc' for i in range(len(best_val_class_performance))] + \
    ['epochs', 'initial_lr', 'best_val_loss', 'lr_schedule']

    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        row = {
            'id': id,
            'model': config['model'],
            'target_column': config['target_column'],
            'additional_columns': json.dumps(config['additional_columns']),  # Use JSON to store list
            'balance_dataset': config['balance_dataset'],
            'use_augmentation': config['use_augmentation'],
            'use_quantized': config['use_quantized'],
            'val_ratio': config['val_ratio'],
            'batch_size': config['batch_size'],
            'dropout_rate': config['dropout_rate'],
            'learning_rate': config['learning_rate'],
            'lr_factor': config['lr_factor'],
            'lr_patience': config['lr_patience'],
            'freeze_layers': config['freeze_layers'],
            'num_epochs': config['num_epochs'],
            'use_existing_split': config['use_existing_split'],
            'early_stopping_patience': config['early_stopping_patience'],
            'weight_decay': config['weight_decay'],
            'val_loss': best_val_loss,
            'val_acc': best_val_acc,
        }
        
        # Add per-class accuracy
        for i, acc in enumerate(best_val_class_performance):
            row[f'class_{i}_acc'] = acc
        
        # Add training parameters
        row.update(training_params)
        
        # Convert lr_schedule to string to avoid issues with CSV writing
        row['lr_schedule'] = json.dumps(row['lr_schedule']).replace('"', '""')
        
        writer.writerow(row)

    print(f"Results recorded in {file_path}")

def load_config_from_results(config, results_file, model_id=None):
    csv.register_dialect('custom', delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f, dialect='custom')
        rows = list(reader)

    if model_id:
        result = next((row for row in rows if row['id'] == model_id), None)
        if result is None:
            raise ValueError(f"No model found with id {model_id}")
    else:
        result = rows[-1]  # Get the last row

    # Update config with values from the results file
    config.update({
        'model': result['model'],
        'target_column': result['target_column'],
        'additional_columns': json.loads(result['additional_columns']) if result['additional_columns'] else [],
        'balance_dataset': result['balance_dataset'] == 'True',
        'use_augmentation': result['use_augmentation'] == 'True',
        'use_quantized': result['use_quantized'] == 'True',
        'batch_size': int(float(result['batch_size'])),
        'dropout_rate': float(result['dropout_rate']),
        'weight_decay': float(result['weight_decay']),
    })
    
    # Load training parameters
    training_params = {
        'epochs': int(float(result['epochs'])),
        'initial_lr': float(result['initial_lr']),
        'lr_schedule': json.loads(result['lr_schedule'].replace('""', '"'))
    }
    
    return config, training_params