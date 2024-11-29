import contextlib
import sys
from PIL import Image
import os

import torch
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.models import ResNeXt50_32X4D_Weights
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import linalg


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import precision_recall_fscore_support
from iresnet100 import iresnet100

from LossFunctions import PearsonCorrelationLoss, SignSensitiveMSELoss

class AffectNetDataset(Dataset):
    def __init__(self, img_dir, annotations_dir, transform=None, target_transform=None, indices=None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.target_transform = target_transform
        

        all_image_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        self.labels = [int(np.load(os.path.join(self.annotations_dir, f.split('.')[0] + "_exp.npy"), allow_pickle=True)) for f in all_image_names]


        if indices is not None:
            self.image_names = [all_image_names[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
        else:
            self.image_names = all_image_names


    def __len__(self):
        return len(self.image_names)
    
    def get_rgb_values(self):

        mu_rgb = np.zeros(3)
        std_rgb = np.zeros(3)
        n_images = len(self.image_names)

        for img_name in self.image_names:
            img_path = os.path.join(self.img_dir, img_name)
            img = np.array(Image.open(img_path)) / 255.0
            mu_rgb += img.mean(axis=(0, 1))
            std_rgb += img.std(axis=(0, 1))

        mu_rgb /= n_images
        std_rgb /= n_images
        print(mu_rgb, std_rgb)
        return mu_rgb, std_rgb

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_names[idx])
        image = Image.open(img_name)

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        if self.target_transform and label in [4, 5, 7]:  # Example classes to augment
            image = self.target_transform(image)

        base_name = self.image_names[idx].split('.')[0]
        aro = torch.tensor(np.load(os.path.join(self.annotations_dir, f"{base_name}_aro.npy"), allow_pickle=True).astype(np.float32))
        exp = torch.tensor(int(np.load(os.path.join(self.annotations_dir, f"{base_name}_exp.npy"), allow_pickle=True)))
        lnd = torch.tensor(np.load(os.path.join(self.annotations_dir, f"{base_name}_lnd.npy"), allow_pickle=True).astype(np.float32))
        val = torch.tensor(np.load(os.path.join(self.annotations_dir, f"{base_name}_val.npy"), allow_pickle=True).astype(np.float32))


        return image, (aro, exp, lnd, val)

class MultiOutputModel(nn.Module):
    def __init__(self, base_model):
        super(MultiOutputModel, self).__init__()
        # Assuming the last layer's output features are accessible like this (you need to verify this based on your base model architecture)
        self.in_features = base_model.fc.in_features
        # Arcface layer
        self.arcface = ArcFace(self.in_features, cout=8, s=5000, m=100)
        # Assume the base model's fully connected layer is not directly usable for multi-output:
        self.base_model = nn.Sequential(*list(base_model.children())[:-2])
        # Adding specific layers for each task
        self.aro = nn.Linear(self.in_features, 1)  # Arousal
        #self.exp = nn.Linear(in_features, 8)
        self.lnd = nn.Linear(self.in_features, 136)  # Landmarks (assuming 68 points x 2 coordinates)
        self.val = nn.Linear(self.in_features, 1)  # Valence
    def forward(self, x, labels=None, return_embeddings=False):
        if len(x) == 5:
            a, b, c, d, e = x.size()
            result = self.base_model(x.view(-1, c, d, e))
            x = result.view(a, b, -1).mean(1)
        else:
            x = self.base_model(x)

        x = x.view(x.size(0), -1)

        if return_embeddings:
            return x

        exp_pred = self.arcface(x, labels)

        return self.aro(x), exp_pred, self.lnd(x), self.val(x)

class ArcFace(nn.Module):
    def __init__(self, cin, cout, s=5000, m=100):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.cout = cout
        self.fc = nn.Linear(cin, cout, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)  # Equivalent to Glorot uniform initializer

    def forward(self, feature_vec, ground_truth_vec=None):
        epsilon = 1e-8
        norm_x = torch.norm(feature_vec, dim=1, keepdim=True) + epsilon
        norm_W = torch.norm(self.fc.weight, dim=0, keepdim=True) + epsilon

        x = feature_vec / norm_x
        W = self.fc.weight / norm_W
        
        cos_theta = torch.mm(x, W.T)
        if ground_truth_vec is not None:
            mask = F.one_hot(ground_truth_vec, num_classes=self.cout).float()
            inv_mask = 1. - mask
            
            invalid_mask = (cos_theta < -1.0) | (cos_theta > 1.0)
            theta = torch.acos(torch.clamp(cos_theta, -.999, .999))
            theta_class = theta * mask  # increasing angle theta of the class x belongs to alone
            theta_class_added_margin = theta_class + self.m
            theta_class_added_margin = theta_class_added_margin * mask
            cos_theta_margin = torch.cos(theta_class_added_margin)
            s_cos_t = self.s * cos_theta_margin
            s_cos_j = self.s * cos_theta * inv_mask
            output = s_cos_t + s_cos_j
        else:
            output = cos_theta

        return output


batch_size = 256


torch.autograd.set_detect_anomaly(True)


print(f'Start of Learning Rate Search (Batch size {batch_size})', '\n')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((112, 112)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

augment_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=2, translate=(0.1, 0.1)),

])

dataset_path = 'train_set'
img_dir = os.path.join(dataset_path, 'images')
annotations_dir = os.path.join(dataset_path, 'annotations')
full_dataset = AffectNetDataset(img_dir=img_dir, annotations_dir=annotations_dir, transform=transform, target_transform=augment_transforms)

# Split indices
dataset_size = len(full_dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)
split = int(np.floor(0.2 * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]

# Create datasets
train_dataset = AffectNetDataset(img_dir=img_dir, annotations_dir=annotations_dir, transform=transform, target_transform=augment_transforms, indices=train_indices)
val_dataset = AffectNetDataset(img_dir=img_dir, annotations_dir=annotations_dir, transform=transform, indices=val_indices)

# Create sampler for training dataset
labels = [train_dataset.labels[i] for i in range(len(train_dataset))]
class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
weights = 1. / class_sample_count
samples_weights = np.array([1 * weights[t] if t == 0 else 1 * weights[t] if t == 1 else weights[t] for t in labels])

samples_weights = torch.from_numpy(samples_weights).double()

sampler = WeightedRandomSampler(samples_weights, len(samples_weights))


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=5, sampler=sampler)
validation_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=5)

# Loss functions
mse_loss = nn.MSELoss()
sign_sensitive_mse_loss = SignSensitiveMSELoss()
pearson_loss = PearsonCorrelationLoss()
ce_loss = nn.CrossEntropyLoss()

# Setting up weighted CE Loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = weights / np.sum(weights) * len(np.unique(labels))
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

class_correct = [0] * 8
class_total = [0] * 8

training_losses = []
training_accuracies = []
validation_losses = []
validation_accuracies = []

best_loss = float('inf')  # Initialize with infinity
for i in range(1):
    # grab a new model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_model = iresnet100(pretrained=False) 

        weights_path = "/scratch/cluster/dylantw/arcfacebackbone.pth"
        base_model.load_state_dict(torch.load(weights_path, map_location=device))

        model = MultiOutputModel(base_model).to(device)
    except Exception as e:
        print(f"ERROR as {e}")
    # Example of adjusting only specific layers (say you only want to train the 'exp' layer)
    for name, param in model.named_parameters():
        if not any(sub in name for sub in ["arcface"]):  # Freeze everything but the expression classifier
            param.requires_grad = False
     
    learning_rate = .0001

    # Setup optimizer for trainable parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay = .0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.25)

    num_epochs = 40
    print(f'lr = {learning_rate}, num_epochs = {num_epochs}, raw parameters, search # {i+1}, \n')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        for inputs, (aro, exp, lnd, val) in train_loader:
            inputs = inputs.to(device)
            aro, exp, lnd, val = aro.to(device), exp.to(device), lnd.to(device), val.to(device)


            optimizer.zero_grad()

            if len(inputs.size()) == 5:
                bs, ncrops, c, h, w = inputs.size()
                result = model(inputs.view(-1, c, h, w), exp) # fuse batch size and ncrops
                result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
                aro_pred, exp_pred, lnd_pred, val_pred = result_avg
            else:
                aro_pred, exp_pred, lnd_pred, val_pred = model(inputs, exp)


            # Step 2: Apply .squeeze() to each output
            aro_pred = aro_pred.squeeze()
            exp_pred = exp_pred.squeeze()
            lnd_pred = lnd_pred.squeeze()
            val_pred = val_pred.squeeze()


            # Calculate and backpropagate losses accordingly
            loss_aro = sign_sensitive_mse_loss(aro_pred, aro)
            loss_exp = ce_loss(exp_pred, exp)
            loss_lnd = sign_sensitive_mse_loss(lnd_pred, lnd)
            loss_val = sign_sensitive_mse_loss(val_pred, val)
            total_loss = loss_aro + loss_exp + loss_lnd + loss_val
            train_loss += total_loss.item()
            
            
            _, predicted = torch.max(exp_pred, 1)
            correct_train += (predicted == exp).sum().item()
            total_train += exp.size(0)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_accuracy = 100 * correct_train / total_train
            
        model.eval()  # Set model to evaluation mode
        embeddings = []
        labels = []

        validation_loss = 0
        correct_exp = 0  # To track correct predictions for expression classification
        total_exp = 0  # Total expressions
        all_preds_exp = []
        all_labels_exp = []
        with torch.no_grad():
            for inputs, (aro, exp, lnd, val) in validation_loader:
                inputs = inputs.to(device)
                aro, exp, lnd, val = aro.to(device), exp.to(device), lnd.to(device), val.to(device)

                if len(inputs.size()) == 5:
                    bs, ncrops, c, h, w = inputs.size()
                    result = model(inputs.view(-1, c, h, w), exp) # fuse batch size and ncrops
                    result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
                    aro_pred, exp_pred, lnd_pred, val_pred = result_avg
                else:
                    embedding_output = model(inputs, return_embeddings=True)
                    embeddings.append(embedding_output.cpu().numpy())
                    labels.append(exp.cpu().numpy())
                    aro_pred, exp_pred, lnd_pred, val_pred = model(inputs)

                # Step 2: Apply .squeeze() to each output
                aro_pred = aro_pred.squeeze()
                exp_pred = exp_pred.squeeze()
                lnd_pred = lnd_pred.squeeze()
                val_pred = val_pred.squeeze()


                # Calculate losses for each output
                loss_aro = sign_sensitive_mse_loss(aro_pred, aro)
                loss_exp = ce_loss(exp_pred, exp)
                loss_lnd = sign_sensitive_mse_loss(lnd_pred, lnd)
                loss_val = sign_sensitive_mse_loss(val_pred, val)
                total_val_loss = loss_aro + loss_exp + loss_lnd + loss_val
                validation_loss += total_val_loss.item()

                # Track correct predictions for expression
                _, predicted_exp = torch.max(exp_pred, 1)
                correct_exp += (predicted_exp == exp).sum().item()
                total_exp += exp.size(0)

                # For calculating precision, recall, f1-score for expression
                all_preds_exp.extend(predicted_exp.cpu().numpy())
                all_labels_exp.extend(exp.cpu().numpy())

        scheduler.step(validation_loss)

        for param_group in optimizer.param_groups:
            print("Current learning rate: ", param_group['lr'])

        # Convert list to numpy arrays
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)  # embeddings should be a 2D NumPy array
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='Spectral', s=5)
        plt.colorbar(scatter)
        plt.title('UMAP projection of the embeddings')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.savefig('best40plot' + f'{epoch}' + '.png')  # Save the plot as a PNG file
        plt.close()
        # Calculate overall and class-specific precision, recall, and F1-score
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            all_labels_exp, all_preds_exp, average='weighted')
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            all_labels_exp, all_preds_exp, average=None, labels=[0,1,2,3,4,5,6,7])

        # Print overall metrics
        print(f'Epoch {epoch+1}, Training Loss: {train_loss / len(train_loader)}, Accuracy: {train_accuracy}%')
        print(f'Epoch {epoch+1}, Validation Loss: {validation_loss / len(validation_loader)}, Accuracy: {100 * correct_exp / total_exp}%')
        print(f'Overall Precision: {overall_precision}, Recall: {overall_recall}, F1 Score: {overall_f1}')

        # Print class-specific metrics
        for i, label in enumerate([0,1,2,3,4,5,6,7]):
            print(f"Class {label} - Precision: {class_precision[i]}, Recall: {class_recall[i]}, F1: {class_f1[i]}")

        print("\n")

                # After training calculations
        train_loss_avg = train_loss / len(train_loader)
        train_accuracy_percent = 100 * correct_train / total_train
        training_losses.append(train_loss_avg)
        training_accuracies.append(train_accuracy_percent)

        # After validation calculations
        validation_loss_avg = validation_loss / len(validation_loader)
        validation_accuracy_percent = 100 * correct_exp / total_exp
        validation_losses.append(validation_loss_avg)
        validation_accuracies.append(validation_accuracy_percent)

    

    best_loss = validation_loss
    best_model_wts = copy.deepcopy(model.state_dict())
    try:
        # Specify the full path where you want to save the model
        filepath = '/scratch/cluster/dylantw/ArcFace_best40.pth'
        torch.save(best_model_wts, filepath)
        print(f"Model saved successfully at {filepath}")
    except Exception as e:
        print(f"Failed to save the model. Error: {e}")

# Create DataFrame
'''results_df = pd.DataFrame({
    'Training Loss': training_losses,
    'Training Accuracy': training_accuracies,
    'Validation Loss': validation_losses,
    'Validation Accuracy': validation_accuracies
})'''

# Export to CSV
#results_df.to_csv('training_metrics_efficient_augment_jit.csv', index_label='Epoch')

