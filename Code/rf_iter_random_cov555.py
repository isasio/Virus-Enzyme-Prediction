import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

# Hamming distance function to calculate the difference between two genotypes
def hamming_distance(genotype1, genotype2):
    return sum(c1 != c2 for c1, c2 in zip(genotype1, genotype2))

# Loop for 500 independent runs
for i in range(1, 501):
    # Set a unique random seed for each iteration
    random.seed(i)
    np.random.seed(i)
    
    # Set the output file names for each run
    file_output = f'rf_correct_predictions_class_20seq_20_{i}.csv'
    history_output = f'rf_training_history_iter_20seq_20_{i}.csv'
    class_predictions_output = f'class_predictions_run_{i}.dat'
    
    # Variables to control the training process
    initial_training_size = 20
    genotypes_to_add_per_iteration = 20
    iterations = 20
    
    # Load the data
    data = pd.read_csv('lasso_prediction_CoV555_aic_opt_order_biochem.txt', sep='\s+')
    data['Genotype'] = data['Genotype'].astype(str).apply(lambda x: x.zfill(15))
    
    def assign_class(row):
        if pd.isna(row['Actual']):
            return 0
        elif row['Actual'] > 6.0:
            return 1
        else:
            return 0

    data['Class'] = data.apply(assign_class, axis=1)
    data['Actual'] = data.apply(
        lambda row: row['Predicted'] if pd.isna(row['Actual']) and row['Predicted'] <= 6 
        else random.uniform(3, 5.5) if pd.isna(row['Actual']) else row['Actual'], axis=1
    )
    
    initial_length = len(data)
    data = data.dropna(subset=['Class'])
    final_length = len(data)
    data['Class'] = data['Class'].astype(int)
    
    # Convert genotype strings to integer arrays for model training
    X_full = np.array([list(map(int, list(genotype))) for genotype in data['Genotype']])
    y_regressor = data['Actual'].values
    y_classifier = data['Class'].values
    
    # Split data into initial training set and the rest
    X_train_initial, X_rest, y_train_regressor_initial, y_rest_regressor, y_train_classifier_initial, y_rest_classifier = train_test_split(
        X_full, y_regressor, y_classifier, 
        train_size=initial_training_size, random_state=i, stratify=y_classifier
    )
    
    selected_genotypes = X_train_initial
    selected_targets_regressor = y_train_regressor_initial
    selected_targets_classifier = y_train_classifier_initial
    selected_genotype_strings = set(''.join(map(str, row)) for row in selected_genotypes)
    
    predictions_df = pd.DataFrame({'Genotype': data['Genotype'], 'Actual': y_regressor})
    history_data = []
    
    # **Initialize class_predictions_output file**
    class_predictions = pd.DataFrame({'Genotype': data['Genotype'], 'Actual': data['Class']})
    class_predictions.to_csv(class_predictions_output, index=False, sep='\t')
    
    # Iterative training loop
    for iteration in range(iterations):
        num_genotypes_to_add = min(genotypes_to_add_per_iteration, len(X_rest))
        if num_genotypes_to_add <= 0:
            print("No more genotypes to add. Stopping sooner than expected.")
            break
        
        # Train the Regressor
        regressor = RandomForestRegressor(n_estimators=20)
        regressor.fit(selected_genotypes, selected_targets_regressor)

        # Train the Classifier
        classifier = RandomForestClassifier(n_estimators=20)
        classifier.fit(selected_genotypes, selected_targets_classifier)

        # Predictions
        train_predictions_reg = regressor.predict(selected_genotypes)
        train_loss_reg = mean_squared_error(selected_targets_regressor, train_predictions_reg)
        val_predictions_reg = regressor.predict(X_rest)
        val_loss_reg = mean_squared_error(y_rest_regressor, val_predictions_reg)
        
        val_predictions_class = classifier.predict(X_rest)
        val_accuracy_class = accuracy_score(y_rest_classifier, val_predictions_class)
       
        # Record iteration data for history
        iteration_data = {
            'Iteration': iteration + 1,
            'Training Size': len(selected_genotypes),
            'Validation Size': len(X_rest),
            'Training MSE': train_loss_reg,
            'Validation MSE': val_loss_reg,
            'Validation Accuracy': val_accuracy_class
        }
        history_data.append(iteration_data)


        # **Predict class for all genotypes in the dataset**
        iteration_predictions_class = classifier.predict(X_full)
        column_name = f'Iteration_{iteration + 1}'
        class_predictions[column_name] = iteration_predictions_class
        class_predictions.to_csv(class_predictions_output, index=False, sep='\t')
        
        # Select genotypes to add to the training set with a Hamming distance of at least 3
        selected_for_training = set()
        max_attempts = 100000  # Limit the number of attempts to avoid infinite loops
        attempts = 0
        hamming_distance_threshold = 3  # Initial threshold for Hamming distance
        
        while len(selected_for_training) < num_genotypes_to_add and attempts < max_attempts:
            # Randomly select a genotype from the validation set
            rand_index = np.random.choice(len(X_rest))
            low_genotype = ''.join(map(str, X_rest[rand_index]))
            
            # Check Hamming distance against all genotypes in the current training set
            if all(hamming_distance(low_genotype, ''.join(map(str, gen))) >= hamming_distance_threshold for gen in selected_genotypes):
                selected_for_training.add(low_genotype)
                
            attempts += 1
        
        # If the required genotypes were not added due to the Hamming distance condition,
        # relax the Hamming distance threshold to 2 and keep searching
        if len(selected_for_training) < num_genotypes_to_add:
            attempts = 0
            print(f"Iteration {iteration + 1}: Could not add 20 genotypes with threshold {hamming_distance_threshold}. Lowering threshold to 2.")
            hamming_distance_threshold = 2  # Lower the threshold to 2
            while len(selected_for_training) < num_genotypes_to_add and attempts < max_attempts:
                # Randomly select a genotype from the validation set
                rand_index = np.random.choice(len(X_rest))
                low_genotype = ''.join(map(str, X_rest[rand_index]))
                
                # Check Hamming distance against all genotypes in the current training set
                if all(hamming_distance(low_genotype, ''.join(map(str, gen))) >= hamming_distance_threshold for gen in selected_genotypes):
                    selected_for_training.add(low_genotype)
                
                attempts += 1
        
        # If still not enough genotypes, forcefully select the remaining ones
        if len(selected_for_training) < num_genotypes_to_add:
            remaining_to_add = num_genotypes_to_add - len(selected_for_training)
            remaining_genotypes = set(''.join(map(str, row)) for row in X_rest)
            selected_for_training.update(list(remaining_genotypes)[:remaining_to_add])
        
        # Convert the genotype string in selected_for_training to integer arrays
        selected_indices_for_iteration = []
        for selected_genotype in selected_for_training:
            selected_genotype_array = np.array(list(map(int, selected_genotype)))  # Convert string to array of integers
            # Find the index of the corresponding genotype in X_full
            idx = np.where((X_full == selected_genotype_array).all(axis=1))[0]
            if len(idx) > 0:
                selected_indices_for_iteration.append(idx[0])  # Append the first matching index

        # Now selected_indices_for_iteration contains the correct indices
        selected_indices_for_iteration = [index for index in selected_indices_for_iteration if index < len(X_rest)]

        if len(selected_indices_for_iteration) > 0:
            X_train_iter = X_rest[selected_indices_for_iteration]
            y_train_regressor_iter = y_rest_regressor[selected_indices_for_iteration]
            y_train_classifier_iter = y_rest_classifier[selected_indices_for_iteration]

            selected_genotypes = np.vstack([selected_genotypes, X_train_iter])
            selected_targets_regressor = np.concatenate([selected_targets_regressor, y_train_regressor_iter])
            selected_targets_classifier = np.concatenate([selected_targets_classifier, y_train_classifier_iter])

            # Remove selected genotypes from the rest set
            X_rest = np.delete(X_rest, selected_indices_for_iteration, axis=0)
            y_rest_regressor = np.delete(y_rest_regressor, selected_indices_for_iteration)
            y_rest_classifier = np.delete(y_rest_classifier, selected_indices_for_iteration)
        
    
    # Save the training history
    pd.DataFrame(history_data).to_csv(history_output, index=False)

