import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import Perceptron

AImodel = KNeighborsClassifier(n_neighbors=1)
# AImodel = svm.SVC()
# AImodel = Perceptron()


TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        evidence = []
        labels = []
        for row in reader:
            evidenceRow = []
            evidenceRow =  row[:17]
            for i in range(len(evidenceRow)):
                if i == 0 or i == 2 or i == 4 or i == 11 or i == 12 or i == 13 or i == 14:
                    evidenceRow[i] = int(evidenceRow[i])
                elif i == 10:
                    if evidenceRow[i] == 'Jan':
                        evidenceRow[i] = 0
                    elif evidenceRow[i] == 'Feb':
                        evidenceRow[i] = 1
                    elif evidenceRow[i] == 'Mar':
                        evidenceRow[i] = 2
                    elif evidenceRow[i] == 'Apr':
                        evidenceRow[i] = 3
                    elif evidenceRow[i] == 'May':
                        evidenceRow[i] = 4
                    elif evidenceRow[i] == 'June':
                        evidenceRow[i] = 5
                    elif evidenceRow[i] == 'Jul':
                        evidenceRow[i] = 6
                    elif evidenceRow[i] == 'Aug':
                        evidenceRow[i] = 7
                    elif evidenceRow[i] == 'Sep':
                        evidenceRow[i] = 8
                    elif evidenceRow[i] == 'Oct':
                        evidenceRow[i] = 9
                    elif evidenceRow[i] == 'Nov':
                        evidenceRow[i] = 10
                    elif evidenceRow[i] == 'Dec':
                        evidenceRow[i] = 11

                elif i == 15:
                    if evidenceRow[i] == 'Returning_Visitor':
                        evidenceRow[i] = 1
                    else:
                        evidenceRow[i] = 0

                elif i == 16:
                    if evidenceRow[i] == 'TRUE':
                        evidenceRow[i] = 1
                    else:
                        evidenceRow[i] = 0
                else:
                    evidenceRow[i] = float(evidenceRow[i])


            evidence.append(evidenceRow)

            if row[-1] == 'TRUE':
                labels.append(1)
            else:
                labels.append(0)
    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # print(evidence[0:10])
    # print(labels[0:10])
    return AImodel.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # print(labels[0:10])
    # print(predictions[0:10])

    countPositive = 0
    countSensitivity = 0
    countNegative = 0
    countSpecificity = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            countPositive += 1
            if predictions[i] == 1:
                countSensitivity += 1
        else:
            countNegative += 1
            if predictions[i] == 0:
                countSpecificity += 1
    Sensitivity = countSensitivity/countPositive
    Specificity = countSpecificity/countNegative

    return (Sensitivity, Specificity)


if __name__ == "__main__":
    main()
