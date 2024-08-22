import csv
import datetime
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    print("returned from loading data")
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
    print("Running Read File")
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        """
        data = []
        for row in reader:
            data.append({
                "evidence": [float(cell) for cell in row[:16]],
                #"label": "Purchased" if row[16] == 'True' else "No Purchase"
            })
        """
        data = []
        labels = []
        for row in reader:
            data.append(row[:17])
            labels.append( 1 if row[17] == "TRUE" else 0)
    # data formating
    for row in data:
        #change administrative to int
        row[0] = int(row[0])
        #change aministrative duration to float
        row[1] = float(row[1])
        # change informational to int
        row[2] = int(row[2])
        # change informational duration to float
        row[3] = float(row[3])
        # change ProductRelated to int
        row[4] = int(row[4])
        # change ProductRelated duration to float
        row[5] = float(row[5])
        # change bounce and exit rates, special day and
        # page values duration to float

        # **** DATA CLEANING **** found a fortran exponent on one piece of
        # data so will clean using a special function
        row[6] = custom_float_conversion(row[6])
        row[7] = float(row[7])
        row[8] = float(row[8])
        row[9] = float(row[9])
        # convert month to int
        # **** DATA CLEANING **** first convert June to Jun
        if row[10] == 'June':
            row[10] = 'Jun'
        row[10] = datetime.datetime.strptime(row[10], "%b").month
        #convert remaining to int
        row[11] = int(row[11])
        row[12] = int(row[12])
        row[13] = int(row[13])
        row[14] = int(row[14])
        # change visitor type to 1 for returning visitor
        # and 0 for all others
        if row[15] == 'Returning_Visitor':
            row[15] = 1
        else:
            row[15] = 0
        # change FALSE to 0 and TRUE to 1 in Weekend
        if row[16] == 'FALSE':
            row[16] = 0
        else:
            row[16] = 1

    print("Finished loading")
    return((data,labels))


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    print("Running Train Model")

    model = KNeighborsClassifier(n_neighbors=1)
    return model.fit(evidence, labels)


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
    print("running Evaluate")
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for x in range(len(labels)):
        if labels[x] == 1:
            if predictions[x] == 1:
                TP += 1
            else:
                FP += 1
        if labels[x] == 0:
            if predictions[x] == 0:
                TN += 1
            else:
                FN += 1
    print("  ")
    print("      Confusing Matrix")
    print("    *********************")
    print(' ')
    print(" ")
    print("        Pos     Neg")
    print("    *********************")
    print(f"Pos *   {TP}   *   {FP}   *")
    print("    *********************")
    print(f"Neg *   {FN}   *  {TN}   *")
    print("    *********************")
    print(" ")

    sensitivity = float(TP/(TP + FN))
    specificity = float(TN/(TN+FP))

    # sensitivity, specifity
    return(sensitivity, specificity)

def custom_float_conversion(s):
    # replace 'd' (fortran exponent) with standard 'e"
    s = s.replace('d', 'e').replace('D', 'E')
    return float(s)

if __name__ == "__main__":
    main()
