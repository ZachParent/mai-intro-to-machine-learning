import pandas as pd
import time

from sklearn.metrics import precision_score, recall_score, f1_score

from tools.SheenaLang.knn import KNNClassifier
from tools.SheenaLang.distance import ManhattanDistance, EuclideanDistance, ChebyshevDistance
from tools.SheenaLang.preprocess import load_datasets, preprocess_mushroom_datasets
from tools.SheenaLang.voting import MajorityClassVote, InverseDistanceWeightedVote, ShepardsWorkVote
from tools.SheenaLang.weighting import InformationGainWeighting, ReliefFWeighting, EqualWeighting

if __name__ == '__main__':
    train_dfs = load_datasets('./datasetsCBR/mushroom/*train.arff')
    test_dfs = load_datasets('./datasetsCBR/mushroom/*test.arff')

    processed_train_dfs = [preprocess_mushroom_datasets(df) for df in train_dfs]
    processed_test_dfs = [preprocess_mushroom_datasets(df) for df in test_dfs]

    k_values = [1, 3, 5, 7]

    weighting_mechanisms = [
        InformationGainWeighting(),
        ReliefFWeighting(),
        EqualWeighting()
    ]

    distance_functions = [
        ManhattanDistance(),
        EuclideanDistance(),
        ChebyshevDistance()
    ]

    voting_schemes = [
        MajorityClassVote(),
        InverseDistanceWeightedVote(),
        ShepardsWorkVote()
    ]

    # Prepare to store results in a DataFrame
    results = []

    # Loop through all combinations of k, distance function, voting scheme, and weighting mechanism
    for k in k_values:
        for distance_func in distance_functions:
            for voting_func in voting_schemes:
                for weighting_mechanism in weighting_mechanisms:

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

                        print(
                            f"Testing combination: k={k}, distance_func={distance_func.__class__.__name__}, "
                            f"voting_func={voting_func.__class__.__name__}, "
                            f"weights={weighting_mechanism.__class__.__name__}"
                        )

                        # Create the KNN classifier with the current parameters and weights
                        knn = KNNClassifier(k=k, distance_func=distance_func, voting_func=voting_func, weights=weights)

                        # Measure accuracy for this pair
                        start_efficiency = time.time()
                        knn.fit(X_train, y_train)  # Fit the model on the training set
                        y_pred = knn.predict(X_test)  # Get predictions
                        score = knn.score(X_test, y_test)  # Test the model on the corresponding test set
                        end_efficiency = time.time()

                        # Update cumulative metrics
                        cumulative_accuracy += score
                        cumulative_precision += precision_score(y_test, y_pred, average='macro', zero_division=0)
                        cumulative_recall += recall_score(y_test, y_pred, average='macro', zero_division=0)
                        cumulative_f1 += f1_score(y_test, y_pred, average='macro', zero_division=0)
                        cumulative_efficiency_time += (end_efficiency - start_efficiency)
                        cumulative_total_time += (time.time() - start_time)

                        print(f"Fold {i+1} Score: {score}")
                        print(f"Fold {i+1} Efficiency Time: {end_efficiency - start_efficiency}")
                        print(f"Fold {i+1} Total Time: {time.time() - start_time}")

                    # Compute average metrics across the 10 folds
                    avg_accuracy = cumulative_accuracy / len(processed_train_dfs)
                    avg_precision = cumulative_precision / len(processed_train_dfs)
                    avg_recall = cumulative_recall / len(processed_train_dfs)
                    avg_f1 = cumulative_f1 / len(processed_train_dfs)
                    avg_efficiency_time = cumulative_efficiency_time / len(processed_train_dfs)
                    avg_total_time = cumulative_total_time / len(processed_train_dfs)

                    print(f"Average Score: {avg_accuracy}")
                    print(f"Average Precision: {avg_precision}")
                    print(f"Average Recall: {avg_recall}")
                    print(f"Average F1 Score: {avg_f1}")
                    print(f"Average Efficiency Time: {avg_efficiency_time}")
                    print(f"Average Total Time: {avg_total_time}")

                    # Store results in the DataFrame format
                    results.append({
                        'k': k,
                        'distance': distance_func.__class__.__name__,
                        'voting': voting_func.__class__.__name__,
                        'weighting': weighting_mechanism.__class__.__name__,
                        'accuracy': avg_accuracy,
                        'precision': avg_precision,
                        'recall': avg_recall,
                        'f1': avg_f1,
                        'efficiency': avg_efficiency_time,
                        'problem-solving time': avg_total_time
                    })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Print the results DataFrame
    print("Final Results DataFrame:")
    print(results_df)

    # Save the DataFrame to a CSV file
    results_df.to_csv('./outputs/knn_results_mushroom.csv', index=False)

    # Find the best combination of parameters based on accuracy
    best_row = results_df.loc[results_df['accuracy'].idxmax()]
    print("Best configuration:")
    print(best_row)

    # Extract the best hyperparameters
    best_k = best_row['k']
    best_distance = best_row['distance']
    best_voting = best_row['voting']
    best_weighting = best_row['weighting']

    # Initialize the final model using the best hyperparameters
    best_distance_func = eval(best_distance)()  # Convert string to class instance
    best_voting_func = eval(best_voting)()      # Convert string to class instance
    best_weighting_mechanism = eval(best_weighting)()  # Convert string to class instance

    # Retrain the model with the entire dataset using the best parameters
    final_cumulative_accuracy = 0
    final_cumulative_precision = 0
    final_cumulative_recall = 0
    final_cumulative_f1 = 0

    for i in range(len(processed_train_dfs)):
        # Get the train and test datasets for this fold
        X_train = processed_train_dfs[i].drop(columns=['class']).to_numpy()
        y_train = processed_train_dfs[i]['class'].to_numpy()

        X_test = processed_test_dfs[i].drop(columns=['class']).to_numpy()
        y_test = processed_test_dfs[i]['class'].to_numpy()

        # Fit the best weighting mechanism on the training set
        best_weighting_mechanism.fit(X_train, y_train)
        weights = best_weighting_mechanism.get_weights()

        # Create the final KNN classifier with the best parameters and weights
        final_knn = KNNClassifier(k=best_k, distance_func=best_distance_func, voting_func=best_voting_func, weights=weights)

        # Train the model on the training set and evaluate on the corresponding test set
        final_knn.fit(X_train, y_train)
        y_pred_final = final_knn.predict(X_test)

        # Calculate metrics
        final_score = final_knn.score(X_test, y_test)
        final_cumulative_accuracy += final_score
        final_cumulative_precision += precision_score(y_test, y_pred_final, average='macro', zero_division=0)
        final_cumulative_recall += recall_score(y_test, y_pred_final, average='macro', zero_division=0)
        final_cumulative_f1 += f1_score(y_test, y_pred_final, average='macro', zero_division=0)

        print(f"Fold {i+1} Final Test Score: {final_score}")

    # Compute the final average metrics across all 10 datasets
    final_avg_accuracy = final_cumulative_accuracy / len(processed_train_dfs)
    final_avg_precision = final_cumulative_precision / len(processed_train_dfs)
    final_avg_recall = final_cumulative_recall / len(processed_train_dfs)
    final_avg_f1 = final_cumulative_f1 / len(processed_train_dfs)

    print(f"Final Average Accuracy on Test Set: {final_avg_accuracy}")
    print(f"Final Average Precision on Test Set: {final_avg_precision}")
    print(f"Final Average Recall on Test Set: {final_avg_recall}")
    print(f"Final Average F1 Score on Test Set: {final_avg_f1}")
