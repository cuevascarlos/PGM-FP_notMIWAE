# Train double-VAE model
'''
import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
'''
import argparse
from utils_data import *
from utils_models import *
from utils_plot import *
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Double VAEs trainer')
    parser.add_argument('--id', type=int, default=17, help='ID of the dataset')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the model file', required=True)
    parser.add_argument('--out_dir', type=str, default='Outputs', help='Output directory')
    args = parser.parse_args()

    if args.model_name is None:
        raise ValueError("You must provide a model name to  a trained model.")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    # Set seeds
    torch.manual_seed(0)
    np.random.seed(0)
    

    # Load data
    X_train, X_train_nan, X_train_missing, mask_train, y_train, X_val, X_val_nan, X_val_missing, mask_val, y_val = load_and_preprocess_data(idx=args.id)
    y_train, y_val = label_encoding(y_train, y_val)
    labels_masking = kmeans_masking(mask_val)

    d = X_train.shape[1]          # Input dimension
    n_hidden = 128
    n_latent = d-1       # Latent space dimension (d - 1)
    n_samples = 20         # Importance samples
    batch_size = 16        # Batch size
    num_iterations = 100000        # Number of iterations
    num_epochs = min(num_iterations // (X_train.shape[0] // batch_size),300)    # Number of epochs
    learning_rate = 1e-3   # Learning rate

    rmse_total = []

    for i in range(5):
        if not os.path.exists(f"{args.out_dir}/iter_{i}"):
            os.makedirs(f"{args.out_dir}/iter_{i}")

        model = DualVAE(d, n_latent, [n_hidden, n_hidden], [n_hidden, n_hidden])
        
        # Train the model
        train_loss_history, val_loss_history = train_2VAE(model, X_train_missing, mask_train, X_val_missing, mask_val, 
                                                batch_size, num_epochs, n_samples, learning_rate)
        train_loss_histories.append(train_loss_history)
        val_loss_histories.append(val_loss_history)

        plot_function_history(train_loss_history, val_loss_history, title = args.out_dir+f"/iter_{i}/loss_history")
        
        # Save the model
        torch.save(model.state_dict(), args.out_dir + f"/iter_{i}/" + args.model_name + ".pth")

        rmse, _ = rmse_imputation_2VAE(X_val, X_val_missing, mask_val, model, nb_samples=1_000)
        print(f"RMSE imputation (Validation): {rmse}")
        rmse_total.append(rmse)

        # Plot latent space
        ploting_latent(X_val_missing, mask_val, model, y_val, nb_samples=1000, components=2, title = f"{args.out_dir}/iter_{i}/Labels")
        ploting_latent(X_val_missing, mask_val, model, y_val, nb_samples=1000, components=3, title = f"{args.out_dir}/iter_{i}/Labels")
        
        ploting_latent(X_val_missing, mask_val, model, labels_masking, nb_samples=1000, components=2, title = f"{args.out_dir}/iter_{i}/Masking")
        ploting_latent(X_val_missing, mask_val, model, labels_masking, nb_samples=1000, components=3, title = f"{args.out_dir}/iter_{i}/Masking")

    print(f"Mean RMSE: {np.mean(rmse_total)}")
    print(f"Std RMSE: {np.std(rmse_total)}")