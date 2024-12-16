import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import argparse

from utils_data import *
from utils_models import *
from utils_plot import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Latent Spaces')
    parser.add_argument('--id', type=int, help='ID of the dataset', required=True)
    parser.add_argument('--model_name', type=str, default=None, help='Name of the model file', required=True)
    parser.add_argument('--out_dir', type=str, default='Outputs', help='Output directory')
    parser.add_argument('--clusters', type=int, default=2, help='Number of clusters for the masking')
    args = parser.parse_args()

    if args.model_name is None:
        raise ValueError("You must provide a model name to load a trained model.")
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Load data
    X_train, X_train_nan, X_train_missing, mask_train, y_train, X_val, X_val_nan, X_val_missing, mask_val, y_val = load_and_preprocess_data(idx=args.id)
    y_train, y_val = label_encoding(y_train, y_val)
    labels_masking = kmeans_masking(mask_val, n_clusters=args.clusters)

    d = X_train.shape[1]          # Input dimension
    n_hidden = 128
    n_latent = d-1       # Latent space dimension (d - 1)
    n_samples = 20         # Importance samples
    batch_size = 16        # Batch size
    num_iterations = 100000        # Number of iterations
    num_epochs = num_iterations // (X_train.shape[0] // batch_size)    # Number of epochs
    learning_rate = 1e-3   # Learning rate

    # Load model
    model_architecture = DualVAE(d, n_latent, [n_hidden, n_hidden], [n_hidden, n_hidden])
    model = load_model(args.model_name, model_architecture)

    # Plot latent space
    ploting_latent(X_val_missing, mask_val, model, y_val, nb_samples=1000, components=2, title = f"{args.out_dir}/Labels")
    ploting_latent(X_val_missing, mask_val, model, y_val, nb_samples=1000, components=3, title = f"{args.out_dir}/Labels")
    ploting_latent(X_val_missing, mask_val, model, labels_masking, nb_samples=1000, components=2, title = f"{args.out_dir}/Masking")
    ploting_latent(X_val_missing, mask_val, model, labels_masking, nb_samples=1000, components=3, title = f"{args.out_dir}/Masking")


    

