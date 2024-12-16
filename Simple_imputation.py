import argparse
from utils_data import *
from utils_models import *
from utils_plot import *
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Double VAEs trainer')
    parser.add_argument('--id', type=int, default=17, help='ID of the dataset')
    parser.add_argument('--out_dir', type=str, default='Outputs', help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    # Set seeds
    np.random.seed(0)
    
    # Load data
    X_train, X_train_nan, X_train_missing, mask_train, y_train, X_val, X_val_nan, X_val_missing, mask_val, y_val = load_and_preprocess_data(idx=args.id)

    # Transform into numpy arrays
    X_val_nan = X_val_nan.numpy()
    X_val = X_val.numpy()
    mask_val = mask_val.numpy()
    
    median_values = []
    mean_values = []
    missforest_values = []
    mice_values = []
    for _ in range(5):
        _, rsme_median = median_imputation(X_val_nan, X_val, mask_val)
        _, rsme_mean = mean_imputation(X_val_nan, X_val, mask_val)
        _, rsme_missforest = missforest_imputation(X_val_nan, X_val, mask_val)
        _, rsme_mice = mice_imputation(X_val_nan, X_val, mask_val)
        median_values.append(rsme_median)
        mean_values.append(rsme_mean)
        missforest_values.append(rsme_missforest)
        mice_values.append(rsme_mice)
    
    print(f'RMSE Median imputation: {np.mean(median_values)} (+-{np.std(median_values)})')
    print(f'RMSE Mean imputation: {np.mean(mean_values)} (+-{np.std(mean_values)})')
    print(f'RMSE MissForest imputation: {np.mean(missforest_values)} (+-{np.std(missforest_values)})')
    print(f'RMSE MICE imputation: {np.mean(mice_values)} (+-{np.std(mice_values)})')

    with open(f'{args.out_dir}/imputation_results.txt', 'w') as f:
        f.write(f'RMSE Median imputation: {np.mean(median_values)} (+-{np.std(median_values)})\n')
        f.write(f'RMSE Mean imputation: {np.mean(mean_values)} (+-{np.std(mean_values)})\n')
        f.write(f'RMSE MissForest imputation: {np.mean(missforest_values)} (+-{np.std(missforest_values)})\n')
        f.write(f'RMSE MICE imputation: {np.mean(mice_values)} (+-{np.std(mice_values)})\n')