import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from work2.svm import SVMClassifier
from work2.tools.preprocess import load_datasets, preprocess_hepatitis_datasets

if __name__ == '__main__':
    # Load and preprocess the datasets
    train_dfs = load_datasets('./datasetsCBR/hepatitis/*train.arff')
    test_dfs = load_datasets('./datasetsCBR/hepatitis/*test.arff')

    processed_train_dfs = [preprocess_hepatitis_datasets(df) for df in train_dfs]
    processed_test_dfs = [preprocess_hepatitis_datasets(df) for df in test_dfs]

    kernels = ["linear", "poly", "rbf", "sigmoid"]
    results = []
    best_kernel = None
    best_score = 0

    # Evaluate different kernels
    for kernel in kernels:
        print(f"Evaluating kernel: {kernel}")
        cumulative_accuracy = 0
        cumulative_precision = 0
        cumulative_recall = 0
        cumulative_f1 = 0
        cumulative_efficiency_time = 0
        cumulative_total_time = 0

        # Loop through each dataset pair for training and testing
        for i in range(len(processed_train_dfs)):
            # Get the train and test datasets for this fold
            X_train = processed_train_dfs[i].drop(columns=['Class']).to_numpy()
            y_train = processed_train_dfs[i]['Class'].to_numpy()

            X_test = processed_test_dfs[i].drop(columns=['Class']).to_numpy()
            y_test = processed_test_dfs[i]['Class'].to_numpy()

            # Start timing for problem-solving time
            start_time = time.time()

            # Initialize and fit the SVM classifier
            svm = SVMClassifier(kernel=kernel)
            start_efficiency = time.time()
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            end_efficiency = time.time()

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            # Efficiency and total time
            efficiency_time = end_efficiency - start_efficiency
            total_time = time.time() - start_time

            # Update cumulative values
            cumulative_accuracy += accuracy
            cumulative_precision += precision
            cumulative_recall += recall
            cumulative_f1 += f1
            cumulative_efficiency_time += efficiency_time
            cumulative_total_time += total_time

            print(f"Fold {i + 1}: Accuracy = {accuracy}, Precision = {precision}, Recall = {recall}, F1 = {f1}")

        # Calculate averages over the folds
        avg_accuracy = cumulative_accuracy / len(processed_train_dfs)
        avg_precision = cumulative_precision / len(processed_train_dfs)
        avg_recall = cumulative_recall / len(processed_train_dfs)
        avg_f1 = cumulative_f1 / len(processed_train_dfs)
        avg_efficiency_time = cumulative_efficiency_time / len(processed_train_dfs)
        avg_total_time = cumulative_total_time / len(processed_train_dfs)

        # Store results for this kernel
        results.append({
            'kernel': kernel,
            'accuracy': avg_accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'efficiency': avg_efficiency_time,
            'problem-solving time': avg_total_time
        })

        print(f"Kernel: {kernel}, Avg Accuracy: {avg_accuracy}, Avg Precision: {avg_precision}, Avg Recall: {avg_recall}, Avg F1: {avg_f1}, Avg Efficiency: {avg_efficiency_time}, Avg Total Time: {avg_total_time}")

        # Track the best kernel
        if avg_accuracy > best_score:
            best_score = avg_accuracy
            best_kernel = kernel

    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv('./outputs/svm_results_hepatitis.csv', index=False)

    # Identify and print the best kernel
    print(f"Best Kernel: {best_kernel} with Accuracy: {best_score}")

    # Train final model with the best kernel
    final_svm = SVMClassifier(kernel=best_kernel)
    X_train_final = np.concatenate([processed_train_dfs[i].drop(columns=['Class']).to_numpy() for i in range(len(processed_train_dfs))], axis=0)
    y_train_final = np.concatenate([processed_train_dfs[i]['Class'].to_numpy() for i in range(len(processed_train_dfs))], axis=0)
    X_test_final = np.concatenate([processed_test_dfs[i].drop(columns=['Class']).to_numpy() for i in range(len(processed_test_dfs))], axis=0)
    y_test_final = np.concatenate([processed_test_dfs[i]['Class'].to_numpy() for i in range(len(processed_test_dfs))], axis=0)

    final_svm.fit(X_train_final, y_train_final)
    y_pred_final = final_svm.predict(X_test_final)

    # Evaluate final model
    final_accuracy = accuracy_score(y_test_final, y_pred_final)
    final_precision = precision_score(y_test_final, y_pred_final, average='macro', zero_division=0)
    final_recall = recall_score(y_test_final, y_pred_final, average='macro', zero_division=0)
    final_f1 = f1_score(y_test_final, y_pred_final, average='macro', zero_division=0)

    print(f"Final Model with {best_kernel} Kernel: Accuracy = {final_accuracy}, Precision = {final_precision}, Recall = {final_recall}, F1 = {final_f1}")