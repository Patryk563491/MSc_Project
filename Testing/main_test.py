##########################################################################################################
# IMPORT MODULES

# Standard libaries
import os
import torch
import numpy as np
import pandas as pd
import random
import sys
import glob
import json

# SKLearn Modules
from sklearn.model_selection import KFold

# MONAI libraries
import monai
from monai.utils import set_determinism
from monai.networks.nets import UNet
from monai.data import CacheDataset, DataLoader
from monai.metrics import DiceMetric, SurfaceDistanceMetric, HausdorffDistanceMetric
from monai.transforms import (Compose,
                              AsDiscreted,
                              AsDiscrete,
                              LoadImaged, 
                              EnsureChannelFirstd, 
                              ToTensord, 
                              ScaleIntensityRanged, 
                              EnsureTyped)



##########################################################################################################
# SET UP DIRECTORIES AND VARIABLES


# Model testing variables
threshold = 0.5 # Threshold for binarisation during validation metric calculation
num_classes = 2 # Number of prediction classes

# Main model directory
main_prefix_dir = r"A-2-2" # CHANGE IF MODEL HYPERPARAMETERS CHANGED

# Testing image and label directory
#testing_directory = r"/jmain02/home/J2AD014/mtc11/ppw59-mtc11/testing_samples/"
testing_directory = r"/jmain02/home/J2AD014/mtc11/ppw59-mtc11/testing_samples/"

# Model state and metrics directory
model_state_dir = os.path.join(main_prefix_dir, "model_state")

test_metrics_save_dir = os.path.join(main_prefix_dir, "metrics")
test_metrics_save_file = os.path.join(test_metrics_save_dir, "test_metrics_majority_vote.csv")

summary_metrics_save_file = os.path.join(test_metrics_save_dir, "summary_metrics_majority_vote.csv")


# Set the random seed for reproducibility
random_seed = 42
random.seed(random_seed)

torch.manual_seed(0)
set_determinism(seed = 0)



##########################################################################################################
# DEFINE REQUIRED FUNCTIONS AND CLASSES

# Define required functions and classes - excludes testing functions

def check_directories():
    """
    Checks if required directories exist; done in two separate steps: data directories and model state/output directories:
        - For data directories, if directories do not exist, OSError is raised.
        - For model state/output directories, if directories do not exist, new directories are created.
    """
    # Check data directories exist
    print() 
    
    for directory in [testing_images_dir, testing_labels_dir]:
        if os.path.exists(directory):
            print(f"The directory '{directory}' exists\n")
        elif not os.path.exists(directory):
            raise FileNotFoundError(f"The directory {directory} does not exist.\n")
    
    # Check directories to save model state and model inference output exist - if not, create new directories
    for directory in [model_state_dir, test_metrics_save_dir]:
        if os.path.exists(directory):
            print(f"The directory '{directory}' exists\n")
        elif not os.path.exists(directory):
            try:
                os.mkdir(directory)
                print(f"Created directory: {directory}\n")
            except OSError as error:
                print(f"Error creating directory {directory}: {error}\n")
        else:
            raise OSError("Unexpected condition encountered while checking the directory\n")



def create_model():
    """
    Function to create a new model. Used in the testing loop to create a new model for each fold.

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
        combined_df = new_df.drop_duplicates(subset = ["filename"], keep = "last").reset_index(drop = True)
        combined_df = combined_df.sort_values(by = ["filename"])
        #new_df = metrics_df.loc[~metrics_df.index.isin(existing_df.index)]
        #combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # Else, if the CSV file doesn't exist, only add the new metrics
        combined_df = metrics_df
    # Save to CSV
    combined_df.to_csv(filename, index=False)
    print(f"\nMetrics saved to {filename}")



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



def free_up_memory():
    """
    Empties cache. 

    Used after each epoch and fold for efficient memory management.
    """
    torch.cuda.empty_cache()
    gc.collect()



def aggregate_metrics(metrics_df):
    """
    Aggregates the metrics across all patches in the test dataset.

    Parameters:
        - metrics_df (pd.DataFrame): DataFrame containing the metrics for each test sample.

    Returns:
        - summary_metrics (pd.DataFrame): DataFrame containing the aggregated metrics (mean and std).
    """
    # Exclude non-numeric columns like 'filename'
    numeric_columns = metrics_df.select_dtypes(include=[np.number]).columns
    summary_metrics = metrics_df[numeric_columns].agg(['mean', 'std']).transpose()
    
    print("\nAggregated Metrics:")
    print(summary_metrics)

    return summary_metrics



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



##########################################################################################################
# LOAD DATA AND DEFINE TRANSFORMS

# Add image and label subdirectories to main directory
testing_images_dir = os.path.join(testing_directory, "images")
testing_labels_dir = os.path.join(testing_directory, "labels")

# Get the paths of the image and label files for training and testing
testing_image_files = sorted(glob.glob(os.path.join(testing_images_dir, "*.nii")))
testing_label_files = sorted(glob.glob(os.path.join(testing_labels_dir, "*.nii")))

print(f"\nNumber of testing images: {len(testing_image_files)}")
print(f"Number of testing labels: {len(testing_label_files)}")

# Create data dictionaries for training images and labels
test_dicts = [{"image": img, "label": lbl, "filename": os.path.basename(img)} for img, lbl in zip(testing_image_files, testing_label_files)]

check_directories() # Check that data and model state/output directories exist 


#Define transforms
test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(keys=["image"], a_min = -57, a_max = 164, b_min = 0.0, b_max = 1.0, clip = True),
    EnsureTyped(keys=["image", "label"]),
    AsDiscreted(keys=["label"], threshold = threshold, to_onehot = num_classes),  # Convert labels to one-hot encoding
    ToTensord(keys=["image", "label"]),
])

# Post transforms unused
post_transforms = Compose([
    AsDiscrete(threshold = threshold)  # Converts probabilities to 0 or 1 based on threshold
])



##########################################################################################################
# TEST THE MODELS

def majority_voting(predictions_list, threshold=0.5):
    """
    Perform majority voting on a list of predictions.

    Parameters:
        - predictions_list (list): List of predicted probabilities from each model.

    Returns:
        - torch.Tensor: Majority-voted prediction.
    """
    stacked_predictions = torch.stack(predictions_list, dim=0)  # Shape: (num_models, batch_size, C, H, W, D)
    majority_voted_predictions = torch.mean(stacked_predictions, dim=0)  # Averaging across models
    majority_voted_predictions = (majority_voted_predictions >= threshold).float()  # Binarize based on threshold
    return majority_voted_predictions


def test_majority_voting(test_dicts, model_files, threshold=0.5):
    """
    Load multiple models, perform testing on all data, and calculate metrics using majority voting.

    Parameters:
        - test_dicts (list): List of dictionaries containing the test data file paths.
        - threshold (float): Threshold for binarization during metric calculation.

    Returns:
        - metrics_df (pd.DataFrame): DataFrame containing the metrics for each test sample.
    """
    # Define metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    surface_metric = SurfaceDistanceMetric(include_background=False, distance_metric="euclidean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean")

    test_ds = CacheDataset(data=test_dicts, transform=test_transforms, cache_rate=1.0, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    # Initialize metrics DataFrame
    metrics_df = pd.DataFrame(columns=[
        "filename", "dice_score", "surface_score", "hausdorff_score", "iou_score", "recall_score"
    ])

    # Reset metrics before testing
    dice_metric.reset()
    surface_metric.reset()
    hausdorff_metric.reset()

    
    models = []

    # Load each model
    for model_file in model_files:
        model = create_model()
        model_save_dir = os.path.join(model_state_dir, model_file)
        
        if os.path.exists(model_save_dir):
            model.load_state_dict(torch.load(model_save_dir))
            model.eval()
            models.append(model)
            print(f"Loaded model {model_save_dir}")
        else:
            raise OSError(f"Model {model_file} not found.")

    total_samples = len(test_loader)

    for i, test_batch_data in enumerate(test_loader, start=1):
        print(f"\nProcessing sample {i}/{total_samples}...")
        test_inputs, test_labels = test_batch_data["image"].to(device), test_batch_data["label"].to(device)
        filename = test_batch_data["filename"][0]

        fold_predictions = []

        # Get predictions from each model
        for model in models:
            with torch.no_grad():
                outputs = model(test_inputs)
                probabilities = torch.sigmoid(outputs)  # Convert to probabilities
                fold_predictions.append(probabilities)

        # Perform majority voting on the predictions from all models
        if len(models) > 1:
            if i == 1: print(f"\nTesting {len(models)} models using majority voting.")
            majority_prediction = majority_voting(fold_predictions, threshold=threshold)
        elif len(models) == 1:
            if i == 1: print(f"\nTesting {len(models)} model without majority voting.")
            majority_prediction = fold_predictions[0]
            majority_prediction = (majority_prediction >= threshold).float()
        else:
            raise Exception(f"\nUnexpected condition in model processing.")

        # Calculate metrics
        dice_metric(y_pred=majority_prediction, y=test_labels)
        surface_metric(y_pred=majority_prediction, y=test_labels)
        hausdorff_metric(y_pred=majority_prediction, y=test_labels)
        iou_score = compute_iou(majority_prediction, test_labels).item()
        recall_score = compute_recall(majority_prediction, test_labels).item()

        # Aggregate the results
        dice_score = dice_metric.aggregate().item()
        surface_score = surface_metric.aggregate().item()
        hausdorff_score = hausdorff_metric.aggregate().item()

        # Reset metrics for the next sample
        dice_metric.reset()
        surface_metric.reset()
        hausdorff_metric.reset()

        # Append the results to the metrics DataFrame
        new_row = pd.DataFrame([{
            "filename": filename,
            "dice_score": dice_score,
            "surface_score": surface_score,
            "hausdorff_score": hausdorff_score,
            "iou_score": iou_score,
            "recall_score": recall_score
        }])
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

        print(f"\nMetrics for {filename}:")
        print(f"Dice Score: {dice_score:.4f}")
        print(f"Surface Score: {surface_score:.4f}")
        print(f"Hausdorff Score: {hausdorff_score:.4f}")
        print(f"IoU Score: {iou_score:.4f}")
        print(f"Recall Score: {recall_score:.4f}")

    save_metrics_to_csv(metrics_df, test_metrics_save_file)

    summary_metrics_df = aggregate_metrics(metrics_df)
    summary_metrics_df.to_csv(summary_metrics_save_file, index=True)
    print(f"\nSummary metrics saved to {summary_metrics_save_file}")

    return metrics_df


# Test all 3 models with majority voting
test_metrics_save_file = os.path.join(test_metrics_save_dir, "all_models_test_metrics_majority_vote.csv")
summary_metrics_save_file = os.path.join(test_metrics_save_dir, "all_models_summary_metrics_majority_vote.csv")
model_files = [r"/jmain02/home/J2AD014/mtc11/ppw59-mtc11/models/3D_U-Net/A-2-2/model_state/model_fold_1.pth", r"/jmain02/home/J2AD014/mtc11/ppw59-mtc11/models/3D_U-Net/A-2-2/model_state/model_fold_2.pth", r"/jmain02/home/J2AD014/mtc11/ppw59-mtc11/models/3D_U-Net/A-2-2/model_state/model_fold_3.pth"]

test_metrics_df = test_majority_voting(test_dicts=test_dicts, model_files=model_files, threshold=threshold)

# Test model 1
test_metrics_save_file = os.path.join(test_metrics_save_dir, "model_1_test_metrics_majority_vote.csv")
summary_metrics_save_file = os.path.join(test_metrics_save_dir, "model_1_summary_metrics_majority_vote.csv")
model_files = [r"/jmain02/home/J2AD014/mtc11/ppw59-mtc11/models/3D_U-Net/A-2-2/model_state/model_fold_1.pth"]

test_metrics_df = test_majority_voting(test_dicts=test_dicts, model_files=model_files, threshold=threshold)

# Test model 2
test_metrics_save_file = os.path.join(test_metrics_save_dir, "model_2_test_metrics_majority_vote.csv")
summary_metrics_save_file = os.path.join(test_metrics_save_dir, "model_2_summary_metrics_majority_vote.csv")
model_files = [r"/jmain02/home/J2AD014/mtc11/ppw59-mtc11/models/3D_U-Net/A-2-2/model_state/model_fold_2.pth"]

test_metrics_df = test_majority_voting(test_dicts=test_dicts, model_files=model_files, threshold=threshold)

# Test model 3
test_metrics_save_file = os.path.join(test_metrics_save_dir, "model_3_test_metrics_majority_vote.csv")
summary_metrics_save_file = os.path.join(test_metrics_save_dir, "model_3_summary_metrics_majority_vote.csv")
model_files = [r"/jmain02/home/J2AD014/mtc11/ppw59-mtc11/models/3D_U-Net/A-2-2/model_state/model_fold_3.pth"]

test_metrics_df = test_majority_voting(test_dicts=test_dicts, model_files=model_files, threshold=threshold)
