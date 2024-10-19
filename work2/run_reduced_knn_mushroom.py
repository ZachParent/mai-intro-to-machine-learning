import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, f1_score

from work2 import KNNClassifier
from work2.tools.distance import ManhattanDistance
from work2.tools.preprocess import load_datasets, preprocess_mushroom_datasets
from work2.tools.voting import MajorityClassVote
from work2.tools.weighting import InformationGainWeighting

if __name__ == '__main__':
    train_dfs = load_datasets('./datasetsCBR/mushroom/*train.arff')
    test_dfs = load_datasets('./datasetsCBR/mushroom/*test.arff')

    processed_train_dfs = [preprocess_mushroom_datasets(df) for df in train_dfs]
    processed_test_dfs = [preprocess_mushroom_datasets(df) for df in test_dfs]

    # TODO adjust
    k_value = 1
    weighting_mechanism = InformationGainWeighting()
    distance_function = ManhattanDistance()
    voting_scheme = MajorityClassVote()

    # Prepare to store results in a DataFrame
    results = []

    # Reduction methods to evaluate
    reduction_methods = {
        "enn",
        "cnn",
        "drop2"
    }

    # Loop through each reduction method
    for reduction_method in reduction_methods:
        # Variables to store cumulative results for 10-fold cross-validation
        cumulative_accuracy = 0
        cumulative_precision = 0
        cumulative_recall = 0
        cumulative_f1 = 0
        cumulative_efficiency_time = 0
        cumulative_total_time = 0

        # Perform 10-fold cross-validation (one dataset for testing, the rest for training)
        for i in range(len(processed_train_dfs)):
            # Ensure the correct pairing between train and test datasets
            X_train = processed_train_dfs[i].drop(columns=['class']).to_numpy()
            y_train = processed_train_dfs[i]['class'].to_numpy()

            X_test = processed_test_dfs[i].drop(columns=['class']).to_numpy()
            y_test = processed_test_dfs[i]['class'].to_numpy()

            # Fit the current weighting mechanism
            weighting_mechanism.fit(X_train, y_train)
            weights = weighting_mechanism.get_weights()  # Get the weights

            # Start the timer for problem-solving time
            start_time = time.time()

            # Create the KNN classifier
            knn = KNNClassifier(k=k_value, distance_func=distance_function, voting_func=voting_scheme, weights=weights)

            # Apply the selected reduction method
            if reduction_method == "cnn":
                X_train_reduced, y_train_reduced = knn.condensed_nearest_neighbor(X_train, y_train)
            elif reduction_method == "enn":
                X_train_reduced, y_train_reduced = knn.edited_nearest_neighbor(X_train, y_train)
            elif reduction_method == "drop2":
                X_train_reduced, y_train_reduced = knn.drop2(X_train, y_train)

            # Fit the model on the reduced training set
            knn.fit(X_train_reduced, y_train_reduced)
            y_pred = knn.predict(X_test)  # Get predictions

            # Calculate metrics
            score = knn.score(X_test, y_test)  # Assuming you have a score method
            end_time = time.time()

            # Update cumulative metrics
            cumulative_accuracy += score
            cumulative_precision += precision_score(y_test, y_pred, average='macro', zero_division=0)
            cumulative_recall += recall_score(y_test, y_pred, average='macro', zero_division=0)
            cumulative_f1 += f1_score(y_test, y_pred, average='macro', zero_division=0)
            cumulative_efficiency_time += (end_time - start_time)
            cumulative_total_time += (end_time - start_time)

            print(f"Fold {i + 1} Score: {score}")

        # Compute average metrics across the 10 folds
        avg_accuracy = cumulative_accuracy / len(processed_train_dfs)
        avg_precision = cumulative_precision / len(processed_train_dfs)
        avg_recall = cumulative_recall / len(processed_train_dfs)
        avg_f1 = cumulative_f1 / len(processed_train_dfs)
        avg_efficiency_time = cumulative_efficiency_time / len(processed_train_dfs)

        # Store results in the DataFrame format
        results.append({
            'k': k_value,
            'distance': distance_function.__class__.__name__,
            'voting': voting_scheme.__class__.__name__,
            'weighting': weighting_mechanism.__class__.__name__,
            'reduction': reduction_method,
            'accuracy': avg_accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'efficiency': avg_efficiency_time,
            'problem-solving time': cumulative_total_time
        })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Print the results DataFrame
    print("Final Results DataFrame:")
    print(results_df)

    # Save the DataFrame to a CSV file
    results_df.to_csv('./outputs/knn_results_reduction.csv', index=False)

    # Find the best reduction method based on the highest accuracy
    best_result = results_df.loc[results_df['accuracy'].idxmax()]
    best_reduction = best_result['reduction']
    print(f"Best Reduction Method: {best_reduction}")

    # Train final model with the best reduction method
    best_method = {
        "cnn": knn.condensed_nearest_neighbor,
        "enn": knn.edited_nearest_neighbor,
        "drop2": knn.drop2
    }[best_reduction]

    # Use the entire training dataset for final training
    final_X_train = pd.concat([df.drop(columns=['class']) for df in processed_train_dfs]).to_numpy()
    final_y_train = pd.concat([df['class'] for df in processed_train_dfs]).to_numpy()

    # Apply the best reduction method
    final_X_train_reduced, final_y_train_reduced = best_method(final_X_train, final_y_train)

    # Create the final KNN model
    final_knn = KNNClassifier(k=k_value, distance_func=distance_function, voting_func=voting_scheme, weights=weights)
    final_knn.fit(final_X_train_reduced, final_y_train_reduced)

    # Evaluate the final model on the test set
    final_X_test = pd.concat([df.drop(columns=['class']) for df in processed_test_dfs]).to_numpy()
    final_y_test = pd.concat([df['class'] for df in processed_test_dfs]).to_numpy()
    final_y_pred = final_knn.predict(final_X_test)

    # Calculate final evaluation metrics
    final_accuracy = final_knn.score(final_X_test, final_y_test)
    final_precision = precision_score(final_y_test, final_y_pred, average='macro', zero_division=0)
    final_recall = recall_score(final_y_test, final_y_pred, average='macro', zero_division=0)
    final_f1 = f1_score(final_y_test, final_y_pred, average='macro', zero_division=0)

    print("Final Model Evaluation Metrics:")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall: {final_recall:.4f}")
    print(f"F1 Score: {final_f1:.4f}")
