import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os

# Function to apply one-hot encoding to a codon sequence
def one_hot_encode_codon_sequence(codon_seq):
    # Define the possible nucleotide bases
    nucleotides = ['A', 'C', 'G', 'T']
    
    # Create a 4x9 matrix (4 possible nucleotides, 9 codon positions)
    one_hot_matrix = np.zeros((4, 9), dtype=int)
    
    # Iterate over the 9 positions in the codon sequence
    for i, base in enumerate(codon_seq):
        # Find the index of the base in nucleotides (A, C, G, T)
        base_index = nucleotides.index(base)
        # Set the corresponding one-hot encoding position to 1
        one_hot_matrix[base_index, i] = 1
    
    # Flatten the 4x9 matrix to a 1D array of size 36
    return one_hot_matrix.flatten()

# Load data from fitness_data_wt.csv
data = pd.read_csv('fitness_data_wt.csv')

# Extract the codon sequences (SV column) and the fitness values (m column)
X_full = np.array([one_hot_encode_codon_sequence(codon_seq) for codon_seq in data['SV']])
y_full = data['m'].values  # Using the m column as y_full (fitness values)
codon_combos = data['SV'].values  # Store codon sequences for reference

# Shuffle the dataset. Just for safety reasons since they are sorted alphabetically
def shuffle_data(X_full, y_full, codon_combos):
    combined = np.column_stack((X_full, y_full, codon_combos))
    np.random.shuffle(combined)
    X_full_shuffled = combined[:, :-2]  # Keep all but last two columns for X_full. Python and their weird ways. I miss you, Julia.
    y_full_shuffled = combined[:, -2].astype(float)  # Second last column is y_full
    codon_combos_shuffled = combined[:, -1]  # Last column is codon_combos
    return X_full_shuffled, y_full_shuffled, codon_combos_shuffled

# Shuffle X_full, y_full, and codon_combos
X_full, y_full, codon_combos = shuffle_data(X_full, y_full, codon_combos)

def train_random_forest(X_train, y_train, X_val, y_val, codon_combos_train, codon_combos_val, num_to_add, num_iterations, output_predictions, output_loss, output_y_count, run_idx):
    predictions_dict = {combo: [] for combo in codon_combos}
    y_count_per_iteration = []

    # Add Iteration 0 count: how many y_train values are >= 1.0
    count_gte_1_initial = np.sum(y_train >= 1.0)
    y_count_per_iteration.append(count_gte_1_initial)

    # Save initial iteration data (Iteration 0)
    with open(f'best_iteration_0_xgb_20.dat', 'a') as f_best, open(f'all_iteration_0_xgb_20.dat', 'a') as f_all, open(f'top_20_iteration_0_xgb_20.dat', 'a') as f_top20:
        best_y_0 = np.max(y_train)
        best_combo_0 = codon_combos_train[np.argmax(y_train)]  # Use codon_combos_train here
        f_best.write(f'{best_combo_0},{best_y_0}\n')

        # Write all sequences
        for combo, y_value in zip(codon_combos_train, y_train):
            f_all.write(f'{combo},{y_value}\n')

        # Get the top 20 sequences
        top_20_indices = np.argsort(y_train)[-20:][::-1]  # Top 20 highest values
        for i in top_20_indices:
            f_top20.write(f'{codon_combos_train[i]},{y_train[i]}\n')

    for iteration in range(num_iterations):
        # Train the random forest model
        model = XGBRegressor(n_estimators=20)
        model.fit(X_train, y_train)
        
        # Predict on all X_full (including both training and validation data)
        y_pred_full = model.predict(X_full)

        # Store predictions for each codon sequence in the iteration
        for i, combo in enumerate(codon_combos):
            predictions_dict[combo].append(y_pred_full[i])

        # Save the highest y_train and corresponding combo for the current iteration
        best_y = np.max(y_train)
        best_combo = codon_combos_train[np.argmax(y_train)]
        
        # Write best y_train and combo to best_iteration_i.dat
        with open(f'best_iteration_{iteration + 1}_xgb_20.dat', 'a') as f_best:
            f_best.write(f'{best_combo},{best_y}\n')

        # Write all codon combos and y_train values for the current iteration to all_iteration_i.dat
        with open(f'all_iteration_{iteration + 1}_xgb_20.dat', 'a') as f_all:
            for combo, y_value in zip(codon_combos_train, y_train):
                f_all.write(f'{combo},{y_value}\n')

        # Write top 20 codon combos and y_train values for the current iteration
        with open(f'top_20_iteration_{iteration + 1}_xgb_20.dat', 'a') as f_top20:
            top_20_indices = np.argsort(y_train)[-20:][::-1]  # Top 20 highest values
            for i in top_20_indices:
                f_top20.write(f'{codon_combos_train[i]},{y_train[i]}\n')

        # Predict on the validation set (X_val)
        y_pred_val = model.predict(X_val)
        
        # Select the top num_to_add predictions from the validation set
        top_indices = np.argsort(y_pred_val)[-num_to_add:][::-1]
        X_to_add = X_val[top_indices]
        y_to_add_actual = y_val[top_indices]
        codon_to_add = codon_combos_val[top_indices]

        # Add selected sequences to training data
        X_train = np.vstack((X_train, X_to_add))
        y_train = np.concatenate((y_train, y_to_add_actual))
        codon_combos_train = np.concatenate((codon_combos_train, codon_to_add))

        # Count how many values in y_train are >= 1.0 for this iteration
        count_gte_1 = np.sum(y_train >= 1.0)
        y_count_per_iteration.append(count_gte_1)

        # Remove selected sequences from validation set
        X_val = np.delete(X_val, top_indices, axis=0)
        y_val = np.delete(y_val, top_indices, axis=0)
        codon_combos_val = np.delete(codon_combos_val, top_indices, axis=0)

        # Calculate loss and validation MSE
        train_loss = mean_squared_error(y_train, model.predict(X_train))
        if len(X_val) > 0:
            val_mse = mean_squared_error(y_val, model.predict(X_val))
        else:
            val_mse = float('nan')

        # Write the loss to output_loss file
        with open(output_loss, 'a') as f_loss:
            f_loss.write(f"Run {run_idx}, Iteration {iteration + 1}: Train Loss: {train_loss:.4f}, Validation MSE: {val_mse:.4f}\n")

    # Save the model after each run
    model_filename = f'xgb_model_run{run_idx}_20seq_20iter_codon.joblib'
    joblib.dump(model, model_filename)

    # Write the counts of y_train values >= 1.0 for each iteration (including Iteration 0)
    with open(output_y_count, 'a') as f_count:
        f_count.write(','.join([str(count) for count in y_count_per_iteration]) + '\n')

    # Write predictions for all iterations to the predictions output file
    with open(output_predictions, 'w') as f_pred:
        # Write the header: codon sequence and iteration numbers
        f_pred.write("CodonSequence," + ",".join([f"Iteration_{i+1}" for i in range(num_iterations)]) + "\n")
        
        # Write the predictions for each codon sequence
        for combo, preds in predictions_dict.items():
            preds_str = ",".join([f"{p:.4f}" for p in preds])
            f_pred.write(f"{combo},{preds_str}\n")


def run_multiple_times(num_times, num_start, num_to_add, num_iterations, output_loss, output_y_count):
    # Ensure output files are cleared before running
    if os.path.exists(output_loss):
        os.remove(output_loss)
    if os.path.exists(output_y_count):
        os.remove(output_y_count)

    for run_idx in range(num_times):
        print(f"\n--- Run {run_idx + 1} of {num_times} ---")

        # Initial random split of data
        X_train, X_val, y_train, y_val, codon_combos_train, codon_combos_val = train_test_split(
            X_full, y_full, codon_combos, train_size=num_start, random_state=(run_idx+1))

        # Run the training process
        train_random_forest(X_train, y_train, X_val, y_val, codon_combos_train, codon_combos_val, num_to_add, num_iterations,
                            f'predictions_output_run{run_idx + 1}_xgb_codon_20.csv', output_loss, output_y_count, run_idx + 1)

# Parameters
num_times = 1000  # Number of runs
num_start = 20  # Number of initial training sequences
num_to_add = 20  # Number of sequences to add in each iteration
num_iterations = 20  # Total number of iterations
output_loss = 'aggregated_loss_output_xgb_codon_20.csv'
output_y_count = 'y_train_count_output_xgb_codon_20.csv'

# Run the training process
run_multiple_times(num_times, num_start, num_to_add, num_iterations, output_loss, output_y_count)
