import csv
import sys
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


class Month(Enum):
    JAN = 0
    FEB = 1
    MAR = 2
    APR = 3
    MAY = 4
    JUNE = 5
    JUL = 6
    AUG = 7
    SEP = 8
    OCT = 9
    NOV = 10
    DEC = 11


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
    """
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        evidence_list = []
        label_list = []
        for row in reader:
            evidence = [
                # add Administrative
                int(row[0]),
                # add Administrative_Duration
                float(row[1]),
                # add Informational
                int(row[2]),
                # add Informational_Duration
                float(row[3]),
                # add ProductRelated
                int(row[4]),
                # add ProductRelated_Duration
                float(row[5]),
                # add BounceRates
                float(row[6]),
                # add ExitRates
                float(row[7]),
                # add PageValues
                float(row[8]),
                # add SpecialDay
                float(row[9]),
                # add Month
                Month[row[10].upper()].value,
                # add OperatingSystems
                int(row[11]),
                # add Browser
                int(row[12]),
                # add Region
                int(row[13]),
                # add TrafficType
                int(row[14]),
                # add VisitorType
                1 if row[15] == "Returning_Visitor" else 0,
                # add Weekend
                1 if row[16].upper() == "TRUE" else 0
            ]
            evidence_list.append(evidence)
            label_list.append(1 if row[17].upper() == "TRUE" else 0)

    return evidence_list, label_list


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).
    """
    sensitivity = 0
    number_of_sensitivity_examples = 0
    specificity = 0
    number_of_specificity_examples = 0

    for i in range(len(labels)):
        if labels[i] == 1:
            number_of_sensitivity_examples += 1
            if labels[i] == predictions[i]:
                sensitivity += 1
        else:
            number_of_specificity_examples += 1
            if labels[i] == predictions[i]:
                specificity += 1

    sensitivity /= number_of_sensitivity_examples
    specificity /= number_of_specificity_examples

    return sensitivity, specificity


if __name__ == "__main__":
    main()
