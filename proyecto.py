# Author: Karla Stefania Cruz Muñiz
# Date: 01.09.2024
# Code for Hebb machine learning algorithm.

import numpy as np

# Input data for AND and OR logic gates
input_data = [[1, 1, -1, -1], [1, -1, 1, -1]]
and_data = [1, -1, -1, -1]
or_data = [1, 1, 1, -1]
inputs = 2

def initialize_weights(input_size):
    # Initializing weight for each input feature
    return np.zeros(input_size)

def train_hebb(input_data, sol_data):
    # Training Hebb algorithm with input data and solution data for each logic gate
    weights = initialize_weights(inputs)
    b_weight = 0
    for idx, yi in enumerate(sol_data):
        for i in range(inputs):
            xi = input_data[i][idx]
            weights[i] += xi * yi 
        b_weight += yi
    return weights, b_weight

def predict_hebb(input_data, weights, b_weight):
    # Predicting using the trained weights and bias weight
    predictions = []
    for idx in range(len(input_data[0])):
        weighted_sum = b_weight
        for i in range(inputs):
            weighted_sum += weights[i] * input_data[i][idx]
        prediction = 1 if weighted_sum >= 0 else -1
        predictions.append(prediction)
    return predictions

def calculate_accuracy(predictions, true_labels):
    # Calculating accuracy of the model
    correct_predictions = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct_predictions / len(true_labels)
    return accuracy

def calculate_precision_recall_f1(predictions, true_labels):
    # Calculating precision, recall and F1 score
    tp = sum(1 for p, t in zip(predictions, true_labels) if p == t == 1)
    tn = sum(1 for p, t in zip(predictions, true_labels) if p == t == -1)
    fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == -1)
    fn = sum(1 for p, t in zip(predictions, true_labels) if p == -1 and t == 1)

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score

def confusion_matrix(predictions, true_labels):
    # Calculating confusion matrix
    tp = sum(1 for p, t in zip(predictions, true_labels) if p == t == 1)
    tn = sum(1 for p, t in zip(predictions, true_labels) if p == t == -1)
    fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == -1)
    fn = sum(1 for p, t in zip(predictions, true_labels) if p == -1 and t == 1)
    
    return np.array([[tp, fp], [fn, tn]])

def print_final_equation(weights, b_weight):
    # Printing the final equation of the model
    equation = f"y = {b_weight:.2f}"
    for i, weight in enumerate(weights):
        equation += f" + ({weight:.2f} * x{i+1})"
    print("Final equation of the model:", equation)

def main():
    print("What do you want to do?")
    print("1. Train and predict with AND")
    print("2. Train and predict with OR")
    print("3. Exit")
    option = input("Enter option: ")
    if option == "1":
        # Training Hebb algorithm for AND
        weights_and, b_weight_and = train_hebb(input_data, and_data)
        print("Final weights AND:", weights_and)
        print("Final bias weight AND:", b_weight_and)
        # Print final equation for AND
        print_final_equation(weights_and, b_weight_and)
        # Predictions using the trained weights for AND
        predictions_and = predict_hebb(input_data, weights_and, b_weight_and)
        print("Predictions AND:", predictions_and)
        accuracy_and = calculate_accuracy(predictions_and, and_data)
        precision_and, recall_and, f1_and = calculate_precision_recall_f1(predictions_and, and_data)
        conf_matrix_and = confusion_matrix(predictions_and, and_data)
        print("Accuracy of the AND model:", accuracy_and)
        print("Precision of the AND model:", precision_and)
        print("Recall of the AND model:", recall_and)
        print("F1 Score of the AND model:", f1_and)
        print("Confusion Matrix of the AND model:\n", conf_matrix_and)
    elif option == "2":
        # Training Hebb algorithm for OR
        weights_or, b_weight_or = train_hebb(input_data, or_data)
        print("Final weights OR:", weights_or)
        print("Final bias weight OR:", b_weight_or)
        # Print final equation for OR
        print_final_equation(weights_or, b_weight_or)
        # Predictions using the trained weights for OR
        predictions_or = predict_hebb(input_data, weights_or, b_weight_or)
        print("Predictions OR:", predictions_or)
        accuracy_or = calculate_accuracy(predictions_or, or_data)
        precision_or, recall_or, f1_or = calculate_precision_recall_f1(predictions_or, or_data)
        conf_matrix_or = confusion_matrix(predictions_or, or_data)
        print("Accuracy of the OR model:", accuracy_or)
        print("Precision of the OR model:", precision_or)
        print("Recall of the OR model:", recall_or)
        print("F1 Score of the OR model:", f1_or)
        print("Confusion Matrix of the OR model:\n", conf_matrix_or)
    elif option == "3":
        print("Exiting...")
    else:
        print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
