import time

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    total_train_time = 0.0
    total_test_time = 0.0
    y_trues_all = []
    y_preds_all = []

    # Fit the model
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    total_train_time += train_time

    # Predict on the test set
    start_time = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start_time
    total_test_time += test_time

    # Collect true labels and predictions
    y_trues_all.extend(y_test)
    y_preds_all.extend(y_pred)

    return y_trues_all, y_preds_all, total_train_time, total_test_time