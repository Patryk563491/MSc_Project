##########################################################################################################
# IMPORT MODULES

import os
import glob
import sys
import nibabel as nib # For loading NiFti .nii files
import json
import random
import gc

import numpy as np
import pandas as pd

# SKLearn Modules
from sklearn.model_selection import KFold

# Torch modules
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

from torch.utils.tensorboard import SummaryWriter

# MONAI Modules
import monai
from monai.networks.nets import UNet
from monai.utils import set_determinism
from monai.losses import DiceCELoss, DiceLoss
from monai.data import CacheDataset, DataLoader
from monai.metrics import DiceMetric, SurfaceDistanceMetric, HausdorffDistanceMetric
from monai.transforms import (Compose,
                              AsDiscreted,
                              AsDiscrete,
                              LoadImaged, 
                              EnsureChannelFirstd, 
                              ToTensord, 
                              ScaleIntensityRanged, 
                              EnsureTyped,
                              RandRotate90d,
                              RandZoomd, # Unused
                              RandGaussianNoised,
                              RandGaussianSmoothd,
                              RandAdjustContrastd,
                              RandSpatialCropd, # Unused
                              RandFlipd,
                              RandAffined,
                              Rand3DElasticd # Unused - causes errors on JADE2
                              )



##########################################################################################################
# SET UP DIRECTORIES AND VARIABLES

# Model training variables
lr = 0.0001 # Learning rate for optimizer
loss_function_mode = "DiceCELoss" # Decides which loss function to use. Current modes: DiceCELoss, DiceLoss

k_folds = 3 # Number of folds in k-cross validation
num_epochs = 450 # Number of epochs for training/validation loop
threshold = 0.5 # Threshold for binarisation during validation metric calculation
patience = 30 # Patience (number of epochs) for early stopping
num_classes = 2 # Number of segmentation classes


# Main model directory
main_prefix_dir = r"A-2-3" # CHANGE IF MODEL HYPERPARAMETERS CHANGED

# TensorBoard save directory
tensor_board_dir = os.path.join(main_prefix_dir, "TensorBoard")
# Set up TensorBoard
writer = SummaryWriter(log_dir = tensor_board_dir)

# Training image and label directories
# Change this if directory changes - training and testing data directories must have "images" and "labels" subdirectories
training_directory = r"/jmain02/home/J2AD014/mtc11/ppw59-mtc11/training_samples/"

# Model state and metrics directory - saves model state for each fold, save metrics for cross validation in the same directory as model_state_dir
model_state_dir = os.path.join(main_prefix_dir, "model_state")

fold_indices_file = os.path.join(model_state_dir, "fold_indices.json")

train_metrics_save_dir = os.path.join(main_prefix_dir, "metrics")
train_metrics_save_file = os.path.join(train_metrics_save_dir, "train_metrics_cross_validation.csv")



# Set the random seed for reproducibility
random_seed = 42
random.seed(random_seed)

torch.manual_seed(0)
set_determinism(seed = 0)



##########################################################################################################
# DEFINE REQUIRED FUNCTIONS AND CLASSES

# Define required functions and classes - excludes cross validation function


def check_directories():
    """
    Checks if required directories exist; done in two separate steps: data directories and model state/output directories:
        - For data directories, if directories do not exist, OSError is raised.
        - For model state/output directories, if directories do not exist, new directories are created.
    """
    # Check data directories exist
    print() 
    
    for directory in [training_images_dir, training_labels_dir]:
        if os.path.exists(directory):
            print(f"The directory '{directory}' exists\n")
        elif not os.path.exists(directory):
            raise FileNotFoundError(f"The directory {directory} does not exist.\n")
    
    # Check directories to save model state and model inference output exist - if not, create new directories
    for directory in [model_state_dir, train_metrics_save_dir, tensor_board_dir]:
        if os.path.exists(directory):
            print(f"The directory '{directory}' exists\n")
        elif not os.path.exists(directory):
            try:
                os.mkdir(directory)
                print(f"Created directory: {directory}\n")
            except OSError as error:
                print(f"Error creating directory {directory}: {error}\n")
        else:
            print("Unexpected condition encountered while checking the directory\n")



def create_model():
    """
    Function to create a new MONAI UNet model. Used in the cross validation loop to create a new model for each fold.

    Returns:
        - model (monai.networks.nets.unet.UNet): 3D U-Net model from MONAI.
    """
    model = UNet(
        spatial_dims = 3,
        in_channels = 1,
        out_channels = num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)
    return model



def save_metrics_to_csv(metrics_df, filename):
    """
    Saves metrics to a CSV file. If the file exists, appends the new metrics, ensuring no duplicates are present (based on fold and epoch number).

    Parameters:
        - metrics_df (pd.DataFrame): DataFrame containing the metrics to be saved.
        - filename (str): Path to the CSV file.
    """
    if os.path.exists(filename):
        # If the CSV file already exists, read in the data, combine with the new data, remove duplicates and sort by fold then epoch
        existing_df = pd.read_csv(filename)
        new_df = pd.concat([existing_df, metrics_df])
        combined_df = new_df.drop_duplicates(subset = ["fold", "epoch"], keep = "last").reset_index(drop = True)
        combined_df = combined_df.sort_values(by = ["fold", "epoch"])
        #new_df = metrics_df.loc[~metrics_df.index.isin(existing_df.index)]
        #combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # Else, if the CSV file doesn't exist, only add the new metrics
        combined_df = metrics_df
    # Save to CSV
    combined_df.to_csv(filename, index=False)
    print(f"\nMetrics saved to {filename}")



def save_prediction_image(image, output_path):
    """
    Saves model predictions in NiFti format. 

    Parameters:
        - image (): image to be saved in NiFti format.
        - output_path (): directory to which the image should be saved to. Needs to contain
                          directory and filename. 
    """
    nib.save(nib.Nifti1Image(image, np.eye(4)), output_path)
    print(f"\nSaved prediction image to {output_path}")



def compute_iou(y_pred, y_true, threshold = 0.5):
    """
    Computes the Intersection over Union (IoU) score using the predicted and ground truth labels.
    Applies a threshold to the predicted label values, calculates the intersection and 
    the union, returning the IoU.

    Parameters:
        - y_pred (torch.Tensor): labels predicted by the model.
        - y_true (torch.Tensor): Ground truth labels.
        - threshold (float): Threshold for IoU calculation. Defaults to 0.5.

    Returns:
        - torch.Tensor: intersection over union socre, or 0 if the union is 0. 
    """
    y_pred = (y_pred >= threshold).float()
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return intersection / union if union != 0 else torch.tensor(0.0)



def compute_recall(y_pred, y_true, threshold = 0.5):
    """
    Computes the recall score based on predicted and ground truth labels.

    Parameters:
        - y_pred (torch.Tensor): Labels predicted by the model.
        - y_true (torch.Tensor): Ground truth labels.
        - threshold (float): threshold for recall calculation. Defaults to 0.5.

    Returns:
        - torch.Tensor: recall score or 0 if the sum of true positive and false negative values is 0.
    """
    y_pred = (y_pred >= threshold).float()
    tp = (y_pred * y_true).sum()
    fn = ((1 - y_pred) * y_true).sum()
    return tp / (tp + fn) if (tp + fn) != 0 else torch.tensor(0.0)



class PolyLR(_LRScheduler):
    """
    Polynomial learning rate scheduler. Scheduler adjusts the learning rate according to a polynomial
    decay function, smoothing the training convergence. Inherets _LRScheduler from torch.optim. The
    learning rate is updated as: lr = base_lr * (1 - current_iter / max_iters) ** power.

    Parameters:
        - optimizer (torch.optim.Optimizer): wrapped torch optimizer. 
        - max_iters (int): the maximum number of training epochs.
        - power (float, optional): the power of the polynomial decay. Defaults to 0.9.
        - last_epoch (int, optional): the index of the last epoch. Defaults to -1. 

    Methods:
        - get_lr(): computes the updated learning rate based on the polynomial decay function. 
    """
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1):
        self.max_iters = max_iters
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Computes the updated learning rate based on the polynomial decay function.

        The learning rate is decayed polynomially as training progresses through the epochs. The
        decay is computer as (1 - current_iter / max_iters) ** power.

        Returns:
            - list: a list of updated learning rates for each parameter group. 
        """
        factor = (1 - self.last_epoch / float(self.max_iters)) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]



def load_existing_metrics(filename):
    """
    Loads existing metrics from a CSV file (if they exist) and determines the last completed fold.

    Parameters:
        - filename (str): Path to the CSV file containing the metrics.

    Returns:
        - pd.DataFrame: DataFrame containing the loaded metrics (empty if file does not exist).
        - int: The last completed fold (0 if file does not exist or all folds have been completed or early stopped).
               If the max_epoch for the fold that is returned is less than 300, last_fold - 1 is returned.
        - int: the max_epoch for the highest fold. Returns 0 if the file is empty or all folds have been completed or early stopped. 
        - list: list of all folds that have been marked as having been early stopped. 
    """
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        if not df.empty:
            #df["fold"] = df["fold"].astype(int)
            early_stopped_folds = df[df["early_stopped"] == 1]["fold"].unique().tolist()
            # Determine the last fold and epoch that was not early stopped and has not reached 300 epochs
            for fold in range(1, df["fold"].max() + 1):
                max_epoch = df[df["fold"] == fold]["epoch"].max()
                if max_epoch < num_epochs:
                    print(f"Identified fold {fold} with max_epoch {max_epoch}")
                    print(f"Early stoped folds: {early_stopped_folds}")
                    return df, fold - 2, max_epoch, early_stopped_folds
            return df, 0, 0, early_stopped_folds  # All folds completed or early stopped
    return pd.DataFrame(), 0, 0, []



def free_up_memory():
    """
    Empties cache. 

    Used after each epoch and fold for efficient memory management.
    """
    torch.cuda.empty_cache() # Empties cache
    gc.collect()



##########################################################################################################
# CUDA CHECKS

print(f"Torch version: {torch.__version__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"\nPython version: {sys.version}")

current_device = torch.cuda.current_device()

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(current_device) if torch.cuda.is_available() else "No GPU found")

print("\n\nCurrent device:", torch.cuda.current_device(), "\n\n")

torch.cuda.empty_cache()
if torch.cuda.is_available():
    print(f"Total VRAM available {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    print(f"Memory allocated: {torch.cuda.memory_allocated(current_device) / 1024 ** 2:.2f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved(current_device) / 1024 ** 2:.2f} MB")
    print(f"\nMax memory allocated: {torch.cuda.max_memory_allocated(current_device) / 1024 ** 2:.2f} MB")
    print(f"Max memory cached: {torch.cuda.max_memory_reserved(current_device) / 1024 ** 2:.2f} MB")
    print(f"\nMemory summary: {torch.cuda.memory_stats()}")



##########################################################################################################
# LOAD DATA AND DEFINE TRANSFORMS

# Add image and label subdirectories to main directory
training_images_dir = os.path.join(training_directory, "images")
training_labels_dir = os.path.join(training_directory, "labels")

# Get the paths of the image and label files for training and testing
training_image_files = sorted(glob.glob(os.path.join(training_images_dir, "*.nii")))
training_label_files = sorted(glob.glob(os.path.join(training_labels_dir, "*.nii")))

print(f"\nNumber of training images: {len(training_image_files)}")
print(f"Number of training labels: {len(training_label_files)}")

# Create data dictionaries for training images and labels
train_dicts = [{"image": img, "label": lbl, "filename": os.path.basename(img)} for img, lbl in zip(training_image_files, training_label_files)]

check_directories() # Check that data and model state/output directories exist 

# Shuffle the data_dicts randomly
#random.shuffle(train_dicts)


#Define transforms
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys = ["image", "label"]),
    ScaleIntensityRanged(keys = ["image"], a_min = -57, a_max = 164, b_min = 0.0, b_max = 1.0, clip = True),
    RandRotate90d(keys = ["image", "label"], prob = 0.5, spatial_axes = (0, 1)), # Random rotation
    RandFlipd(keys = ["image", "label"], prob = 0.5, spatial_axis = 0), # Random flip
    RandAffined(keys=["image", "label"], prob = 0.2, rotate_range=(0, 0, 0.1), scale_range=(0.1, 0.1, 0.1)),  # Random scaling
    #Rand3DElasticd(keys = ["image", "label"], sigma_range = (5, 7), magnitude_range = (100, 200), prob = 0.2, spatial_size = (128, 128, 400), mode = ("bilinear", "nearest")),  # Random elastic deformation
    RandGaussianNoised(keys = ["image"], prob = 0.2), # Random Gaussian noise
    RandGaussianSmoothd(keys = ["image"], prob = 0.1), # Random Gaussian smoothing
    RandAdjustContrastd(keys = ["image"], prob = 0.1), # Random adjust contrast
    EnsureTyped(keys = ["image", "label"]),
    AsDiscreted(keys=["label"], threshold = threshold, to_onehot=num_classes),  # Convert labels to one-hot encoding
    ToTensord(keys = ["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(keys=["image"], a_min = -57, a_max = 164, b_min = 0.0, b_max = 1.0, clip = True),
    EnsureTyped(keys=["image", "label"]),
    AsDiscreted(keys=["label"], threshold = threshold, to_onehot=num_classes),  # Convert labels to one-hot encoding
    ToTensord(keys=["image", "label"]),
])

post_transforms = Compose([
    AsDiscrete(threshold=0.5)  # Converts probabilities to 0 or 1 based on threshold
])



##########################################################################################################
# DEFINE AND TRAIN MODEL

# Generate fold indices before starting cross-validation
if not os.path.exists(fold_indices_file):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    fold_indices = [(train_idx.tolist(), val_idx.tolist()) for train_idx, val_idx in kf.split(train_dicts)]
    with open(fold_indices_file, 'w') as file:
        json.dump(fold_indices, file)
    print(f"\nFold indices saved to {fold_indices_file}")
elif os.path.exists(fold_indices_file):
    with open(fold_indices_file, 'r') as file:
        fold_indices = json.load(file)
    print(f"\nFold indices loaded from {fold_indices_file}")
else:
    raise Exception("Error processing KFold indices: Unexpected condition encountered.")

model = create_model()

# Free up memory before starting cross validation
free_up_memory()

def cross_validation(train_dicts, k_folds=5, threshold=0.5, patience=30):
    """
    Performs k-fold cross validation on a given model - training and validation. 

    Training metrics:
        - Training loss

    Validation metrics:
        - Validation loss
        - Dice score
        - Surface score
        - Hausdorff score
    
    Function has several additional features:
        - Prints training and validation metrics at the end of each epoch for each fold.
        - Adds metrics to TensorBoard - requires TensorBoard setup before the loop.
        - Adds the model graph to TensorBoard.
        - On the last epoch, saves model for each fold.
        - On the last epoch of the fold, the model predicts a label, which is saved in NiFti format.
        - After the last epoch of the last fold, all the metrics are saved in a CSV file; columns: fold, epoch, train_loss, val_loss, dice_score, hausdorff_score.

    Requirements:
        - Prior TensorBoard setup.
        - create_model function to crate a new model for each fold.
        - save_metrics_to_csv function to save the metrics for all epochs and folds at the end of the loop.
        - save_prediction_image function to save the model output for each fold.
    
    Parameters:
        - train_dicts (list): List of dictionaries containing the training data file paths.
        - k_folds (int): K number of folds for cross validation. Defaults to 5.
        - threshold (float): Threshold for binarization during metric calculation.
        - patience (int): patience (number of epochs) for early stopping.
    """

    # Loads metrics file if it exists, otherwise returns empty dataframe and 0 as last_fold
    metrics_df, last_fold, last_epoch, early_stopped_folds = load_existing_metrics(train_metrics_save_file)
    print(f"\nLast fold: {last_fold}; last_epoch: {last_epoch}; Early stopped folds: {early_stopped_folds}")
    
    # Define metrics
    dice_metric = DiceMetric(include_background = False, reduction = "mean")
    surface_metric = SurfaceDistanceMetric(include_background=False, distance_metric="euclidean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean")

    for fold in range(k_folds):
        # Skip folds that have already been trained (if applicable)
        if last_fold + 1 > fold:
            print(f"\nSkipping fold {fold + 1} as it has been completed.")
            continue
        """
        # Skip folds that were early stopped
        if early_stopped_folds and fold + 1 in early_stopped_folds:
            print(f"\nSkipping fold {fold + 1} as it was early stopped.")
            continue
        """

        
        # Create model save directory for the fold
        model_save_dir = os.path.join(model_state_dir, f"model_fold_{fold + 1}.pth")
        model = create_model()
        # Load in model from model save directory, if it exists
        if os.path.exists(model_save_dir):
            model.load_state_dict(torch.load(model_save_dir))
            print(f"\nLoaded model state from {model_save_dir}")

        # Assign train_idx and val_idx for the current fold
        train_idx, val_idx = fold_indices[fold]

        print("\n\n")
        print("-" * 60)
        print(f"Fold {fold+1}/{k_folds}")
        print("-" * 60)
        print(f"\nTrain idx: {train_idx}, \nValidation idx: {val_idx}\n")

        # Split the data and create datasets
        train_data = [train_dicts[i] for i in train_idx]
        val_data = [train_dicts[i] for i in val_idx]

        train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0, num_workers=1)
        val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=1.0, num_workers=1)
        
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=1, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=1, persistent_workers=True)

        if loss_function_mode == "DiceCELoss":
            loss_function = DiceCELoss(sigmoid=True, include_background=True)
        elif loss_function_mode == "DiceLoss":
            loss_function = DiceLoss(sigmoid=True, include_background=True)
        else:
            raise Exception("Loss function mode (loss_function_mode) unspecified")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = PolyLR(optimizer=optimizer, max_iters=num_epochs)
        
        """
        if fold == 0 and last_fold == 0:
            example_data = next(iter(train_loader))["image"].to(device)
            writer.add_graph(model, example_data)
        """

        dice_metric.reset()
        surface_metric.reset()
        hausdorff_metric.reset()

        best_val_loss = float("inf")
        no_improvement_count = 0

        # Training and validation loop
        epoch_skip_counter = 0
        for epoch in range(num_epochs):
            # Skip epochs if they have already been trained for this fold
            if last_epoch > epoch and fold == last_fold + 1:
                epoch_skip_counter += 1
                continue
            if epoch_skip_counter != 0 and last_epoch == epoch:
                print(f"\nSkipped {epoch_skip_counter} epochs in fold {fold + 1}, as they were complete.\n")
            
            print("\n", ("-" * 30), sep="")
            print(f"Epoch {epoch + 1}/{num_epochs}")

            model.train()
            train_loss = 0

            # Training loop
            for train_batch_data in train_loader:
                train_inputs, train_labels = train_batch_data["image"].to(device), train_batch_data["label"].to(device)

                optimizer.zero_grad()
                train_outputs = model(train_inputs)
                loss = loss_function(train_outputs, train_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                #free_up_memory()

            train_loss /= len(train_loader)
            print(f"\nTraining loss for epoch {epoch + 1}: {train_loss:.4f}")

            # Validation loop
            model.eval()
            val_loss = 0
            all_iou_scores, all_recall_scores = [], []

            with torch.no_grad():
                for val_batch_data in val_loader:
                    val_inputs, val_labels = val_batch_data["image"].to(device), val_batch_data["label"].to(device)

                    val_outputs = model(val_inputs)
                    loss = loss_function(val_outputs, val_labels)
                    val_loss += loss.item()

                    probabilities = torch.sigmoid(val_outputs)

                    binarized_outputs = torch.where(probabilities >= threshold, torch.ones_like(probabilities), torch.zeros_like(probabilities))

                    dice_metric(y_pred=binarized_outputs, y=val_labels)
                    surface_metric(y_pred=binarized_outputs, y=val_labels)
                    hausdorff_metric(y_pred=binarized_outputs, y=val_labels)
                    all_iou_scores.append(compute_iou(binarized_outputs, val_labels).item())
                    all_recall_scores.append(compute_recall(binarized_outputs, val_labels).item())

            val_loss /= len(val_loader)
            dice_score = dice_metric.aggregate().item()
            surface_score = surface_metric.aggregate().item()
            hausdorff_score = hausdorff_metric.aggregate().item()
            iou_score = np.mean(all_iou_scores)
            recall_score = np.mean(all_recall_scores)

            dice_metric.reset()
            surface_metric.reset()
            hausdorff_metric.reset()
            
            # Add metrics to TensorBoard
            writer.add_scalar(f'Loss/train_fold_{fold + 1}', train_loss, epoch + fold * num_epochs)
            writer.add_scalar(f'Loss/validation_fold_{fold + 1}', val_loss, epoch + fold * num_epochs)
            writer.add_scalar(f'Dice/validation_fold_{fold + 1}', dice_score, epoch + fold * num_epochs)
            writer.add_scalar(f'Surface/validation_fold_{fold + 1}', surface_score, epoch + fold * num_epochs)
            writer.add_scalar(f'Hausdorff/validation_fold_{fold + 1}', hausdorff_score, epoch + fold * num_epochs)
            writer.add_scalar(f"IoU/validation_fold_{fold + 1}", iou_score, epoch + fold * num_epochs)
            writer.add_scalar(f"Recall/validation_fold_{fold + 1}", recall_score, epoch + fold * num_epochs)

            # Print metrics to output file
            print(f"\nValidation loss for epoch {epoch + 1}: {val_loss:.4f}")
            print(f"Validation dice score: {dice_score:.4f}")
            print(f"Validation surface metric score: {surface_score:.4f}")
            print(f"Validation Hausdorff metric score: {hausdorff_score:.4f}")
            print(f"Validation IoU score: {iou_score:.4f}")
            print(f"Validation recall score: {recall_score:.4f}")
            
            # Save metrics in CSV file and model state (done per epoch)
            epoch_metrics = {
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "dice_score": dice_score,
                "surface_score": surface_score,
                "hausdorff_score": hausdorff_score,
                "iou_score": iou_score,
                "recall_score": recall_score,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "early_stopped": 0
            }

            metrics_df = pd.concat([metrics_df, pd.DataFrame([epoch_metrics])], ignore_index=True)
            
            torch.save(model.state_dict(), model_save_dir)
            save_metrics_to_csv(metrics_df, train_metrics_save_file)

            # Early stopping: 1) update the no_improvement counter appropriately; 2) break the for loop for the current fold,
            # if no_improvement_counter reaches patience
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                # If early stop, save model for the fold
                torch.save(model.state_dict(), model_save_dir)
                epoch_metrics["early_stopped"] = 1
                metrics_df = pd.concat([metrics_df, pd.DataFrame([epoch_metrics])], ignore_index=True)
                save_metrics_to_csv(metrics_df, train_metrics_save_file)
                break
            
            # Learning rate scheduler step
            scheduler.step()

            # Free up memory at the end of each epoch
            free_up_memory()

        # Save model and metrics at the end of the fold
        torch.save(model.state_dict(), model_save_dir)
        save_metrics_to_csv(metrics_df, train_metrics_save_file)

        # Free up memory at the end of each fold
        free_up_memory()

    save_metrics_to_csv(metrics_df, train_metrics_save_file)


# Call the function to perform KFold Cross Validation
cross_validation(train_dicts=train_dicts, k_folds=k_folds, threshold=threshold)
