import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Optimize matrix multiplication performance for Ampere GPU architecture
torch.set_float32_matmul_precision("high")

# ------------------- SETTINGS -------------------
data_dir = r"C:\\Darsh_Degree\\Techmanjari 26\\chest_xray"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

num_classes = 3 
batch_size = 16 
num_epochs = 20
learning_rate = 1e-4
max_grad_norm = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = torch.cuda.is_available()
# Utilize multiple CPU cores to prevent data loading bottlenecks
num_workers = 4 if torch.cuda.is_available() else 0

# ------------------- TRANSFORMS -------------------
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# ------------------- DATASETS -------------------
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)

print(f"Classes detected: {train_dataset.classes}")

# Fast class count extraction using targets directly without dataset iteration overhead
class_counts = [0] * num_classes
for label in train_dataset.targets:
    class_counts[label] += 1
    
class_counts = [c if c > 0 else 1 for c in class_counts]
class_weights = [sum(class_counts)/c for c in class_counts]
sample_weights = [class_weights[label] for label in train_dataset.targets]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# ------------------- MODEL CREATION -------------------
def create_model(model_name="densenet121", num_classes=3):
    # Initialize specified backbone network with pre-trained ImageNet weights
    if model_name.lower() == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, num_classes))
    elif model_name.lower() == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, num_classes))
    return model.to(device)

densenet_model = create_model("densenet121", num_classes)
resnet_model = create_model("resnet34", num_classes)
ensemble_models = [densenet_model, resnet_model]

# ------------------- LOSS -------------------
class LabelSmoothingCrossEntropy(nn.Module):
    # Cross entropy loss modified with label smoothing to prevent overfitting
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, pred, target):
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        target_onehot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        target_onehot = target_onehot * (1 - self.smoothing) + self.smoothing / pred.size(1)
        return (-target_onehot * log_probs).sum(dim=1).mean()

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizers = [optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=1e-4) for m in ensemble_models]
schedulers = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs) for opt in optimizers]
scalers = [torch.amp.GradScaler('cuda') for _ in ensemble_models]

# ------------------- TRAINING FUNCTION -------------------
def train_ensemble(models, dataloaders, criterion, optimizers, schedulers, num_epochs=20):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        for phase in ['train', 'val']:
            for model in models:
                model.train() if phase == 'train' else model.eval()

            running_corrects = 0
            total = 0
            pbar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase", ncols=100)

            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs_ensemble = []

                for idx, model in enumerate(models):
                    optimizers[idx].zero_grad()
                    
                    # Compute mixed precision forward and backward passes sequentially per model
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.amp.autocast(device_type='cuda'):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            scalers[idx].scale(loss).backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            scalers[idx].step(optimizers[idx])
                            scalers[idx].update()
                    
                    outputs_ensemble.append(outputs.detach())

                # Average predictions across the ensemble models to track joint performance
                ensemble_outputs = torch.stack(outputs_ensemble).mean(dim=0)
                _, preds = torch.max(ensemble_outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
                
                pbar.set_postfix({"Acc": f"{running_corrects.double()/total:.4f}"})

            epoch_acc = running_corrects.double() / total
            print(f"{phase.capitalize()} Ensemble Accuracy: {epoch_acc:.4f}")

            # Persist model weights to disk if validation accuracy sets a new record
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(models[0].state_dict(), "densenet_best.pt")
                torch.save(models[1].state_dict(), "resnet_best.pt")
                print("--- Saved new best model weights (3-Class) ---")

            if phase == 'val':
                for scheduler in schedulers:
                    scheduler.step()

    return models

# ------------------- TEST FUNCTION -------------------
def test_ensemble(models, loader):
    for m in models: 
        m.eval()
    corrects, total = 0, 0
    
    # Run evaluation with inference mode for maximum runtime speed and lowest memory use
    with torch.inference_mode():
        for inputs, labels in tqdm(loader, desc="Testing", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            preds_list = [torch.softmax(m(inputs), dim=1) for m in models]
            avg_preds = torch.stack(preds_list).mean(dim=0)
            _, final_preds = torch.max(avg_preds, 1)
            corrects += torch.sum(final_preds == labels.data)
            total += labels.size(0)
            
    print(f"\nFinal Test Accuracy: {corrects.double()/total:.4f}")

# ------------------- EXECUTION -------------------
if __name__ == "__main__":
    print(f"Using Device: {device}")
    ensemble_models = train_ensemble(
        ensemble_models, 
        {'train': train_loader, 'val': val_loader}, 
        criterion, 
        optimizers, 
        schedulers, 
        num_epochs=num_epochs
    )
    test_ensemble(ensemble_models, test_loader)
    print("\nTraining Complete. Models can now detect invalid images.")