def accuracy_evaluator(y_pred, y_true):
    correct_predictions = 0
    total_predictions = len(y_true)
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy


def evaluate(predictions, labels):
    accuracy = accuracy_evaluator(predictions, labels)
    return accuracy * 100