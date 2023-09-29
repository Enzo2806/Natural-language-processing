"""
COMP 550 - Assignment 1

Author: Enzo Benoit-Jeannin
McGill ID: 260969262

"""

### IMPORTS
# Pre Processing
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

# Metrics and evaluations
from sklearn.metrics import accuracy_score













### DATA IMPORTATION
print("\nImporting Data...\n")
# Load real facts from facts.txt
with open('./facts.txt', 'r', encoding='utf-8') as real_file:
    X_real = real_file.read().splitlines()

# Load fake facts from fake.txt
with open('./fakes.txt', 'r', encoding='utf-8') as fake_file:
    X_fake = fake_file.read().splitlines()

# Create arrays for real and fake facts and their corresponding labels
y_real = np.zeros(len(X_real))  # We assign label 0 to real facts
y_fake = np.ones(len(X_fake))  # We assign label 1 to fake facts

# Concatenate arrays
X = np.concatenate((X_real, X_fake))
y = np.concatenate((y_real, y_fake))

# Shuffle the data to mix both real and fake data
permutation = np.random.permutation(len(X))
X = X[permutation]
y = y[permutation]

# We now split the data into trainining validation and testing sets
# We choose to split as follows: 70% of the data generated goes into training, 15% into validation and 15% into testing
X_train_unprocessed, X_temp, y_train_unprocessed, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)
X_val_unprocessed, X_test_unprocessed, y_val_unprocessed, y_test_unprocessed = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify = y_temp)

# Safety check: print the length of all split array
# Note that we have 600 examples: 100*3*2 for each animal and both real and fake facts
print("\nArray Size check")
# This should print 420
print("Training set: X length =", len(X_train_unprocessed), ", y length =", len(y_train_unprocessed))
# This should print 90
print("Validation set: X length =", len(X_val_unprocessed), ", y length =", len(y_val_unprocessed))
# This should print 90
print("Testing set: X length =", len(X_test_unprocessed), ", y length =", len(y_test_unprocessed))

# Check the distribution of each set
# The distribution of fakes and real should eb the same in all sets as we set stratisfy when splitting the sets
print("\nDistribution Size check")
print("Training set: Number of fakes =", np.sum(y_train_unprocessed == 0), ", Number of real =", np.sum(y_train_unprocessed == 1))
print("Validation set: Number of fakes =", np.sum(y_val_unprocessed == 0), ", Number of real =", np.sum(y_val_unprocessed == 1))
print("Testing set: Number of fakes =", np.sum(y_test_unprocessed == 0), ", Number of real =", np.sum(y_test_unprocessed == 1))














### DATA PREPROCESSING
class SetType(Enum):
    Train = 1
    Validation = 2
    Testing = 3

# We define a function preprocess that takes as input an X and y matrix.
# The function applies 3 preprocessing techniques seen in class:
# Since lemmatization and stemming are performing the very similar operation of normalising inflected words
# It does not really make sense to apply stemming and lemmatization at the same time.
# Note that if both lemmatization and stemming are set to true, lemmatization is given priority
def preprocess(X, y, vectorizer, setType:SetType, lemmatize=True, stem=False, remove_stopwords=True):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Prepare a list to store preprocessed texts
    X_preprocess = []

    # Apply preprocessing techniques based on input parameters
    for text in X:
        # Split text into words (tokenization)
        # To ensure consistency and reduce dimentionnality of the data, we convert all text to lowercase
        words = text.lower().split()

        # Apply lemmatization if lemmatize is true
        if lemmatize:
            words = [lemmatizer.lemmatize(word) for word in words]

        # Apply stemming only if lemmatization was false and stem = true
        elif stem:
            words = [stemmer.stem(word) for word in words]

        # Remove stop words if remove_stopwords is true
        if remove_stopwords:
            words = [word for word in words if word not in stop_words]

        # Reconstituate the preprocessed text by joining all words separated by a space
        text_preprocessed = ' '.join(words)

        # Add the text to X_preprocess
        X_preprocess.append(text_preprocessed)

    # Transform the preprocessed text data into a numeric feature matrix for the training set:
    if setType == SetType.Train:
        X_vectorized = vectorizer.fit_transform(X_preprocess)
    # The transform function ignores any words not present in the dataset given to the fit function.
    else:
        X_vectorized = vectorizer.transform(X_preprocess)

    return X_vectorized.toarray(), np.array(y)


# Since we want to compare preprocessing techniques, we create a function that returns all sets and the predefined split between training abnd validation sets
# This make comparison easier between models and processing techniques
def getProcessedSets(lemmatize=True, stem=True, remove_stopwords = True):
    print("\nPre-Processing Data...\n")

    # Initialize the count vectorizer to convert text data to numerical feature matrix
    # Use Term Frequency-Inverse Document Frequency Vectorizer:
    # It calculates the frequency of each word in the text and it normalizes the term frequencies by the
    # total number of words in each document. This accounts for varying document lengths.
    # We use the same vectorizer to fit on the training data, then tranform the validation and test sets
    vectorizer = TfidfVectorizer()

    # Preprocess each set independantly to avoid information leakage
    X_train, y_train = preprocess(X_train_unprocessed, y_train_unprocessed, vectorizer=vectorizer, setType=SetType.Train, lemmatize=lemmatize, stem=stem, remove_stopwords = remove_stopwords)
    X_val, y_val = preprocess(X_val_unprocessed, y_val_unprocessed, vectorizer=vectorizer, setType=SetType.Testing, lemmatize=lemmatize, stem=stem, remove_stopwords = remove_stopwords)
    X_test, y_test = preprocess(X_test_unprocessed, y_test_unprocessed, vectorizer=vectorizer, setType=SetType.Validation, lemmatize=lemmatize, stem=stem, remove_stopwords = remove_stopwords)

    # In order to hyperparameter tune each model, we will use the created validation set (and thus not perform cross validation)
    # We therefore specify a predefined split using sklearn
    split_index = np.concatenate([-1 * np.ones(X_train.shape[0]), np.zeros(X_val.shape[0])])
    # We combine both training and validation in one array
    # Note that the GridSearchCV is given the predefined train=validation split so it compares models based on this validation set instead of cross validation
    # for hyperparameter tuning.
    y = np.concatenate([y_train, y_val], axis=0)
    X = np.concatenate([X_train, X_val], axis=0)
    predefined_split = PredefinedSplit(test_fold = split_index)

    return X, y, X_val, y_val, X_test, y_test, predefined_split
















### Model training function definition
# This function hypertunes the given model using the given parameters 
# Retruns the best model, the best parameters and the final validation accuracy (on the best model)
def hypertune(X, y, X_validation, y_validation, classifier, hyperparameters, predefined_split):
    # Use GridSearchCV for hyperparameter tuning
    # the predefined train=validation split is provided so GridSearchCV can compare models based on this validation set 
    # we created instead of doing cross validation
    grid_search = GridSearchCV(classifier, hyperparameters, cv=predefined_split, scoring='accuracy', verbose=1, n_jobs=-1)

    # Fit the model to the training data
    grid_search.fit(X, y)
    
    # Get the best hyperparameters from the grid search
    best_params = grid_search.best_params_
    # Get the best classifier modle obtained
    model_best = grid_search.best_estimator_

    # Predict on the validation data
    y_pred = model_best.predict(X_validation)
    # Calculate validation accuracy for the best model
    best_validation_accuracy = accuracy_score(y_validation, y_pred)

    return model_best, best_params, best_validation_accuracy

# We now define the different classifiers and their corresponding hyperparameters
logistic_regression = LogisticRegression()
logistic_params = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'max_iter': [100, 200]  # Maximum number of iterations
}

naive_bayes = MultinomialNB()
bayes_params = {
    'alpha': [0.1, 1, 10.0], # Laplace smoothing parameter
}

svm = SVC()
svm_params = {
    'C': [0.1, 1, 10], # Regularization parameter
    'kernel': ['linear'], # Try the linear kernel only as it is the only one seen in class 
}

# Array to hold each classifier's information to use when tuning them one after the other
classifiers = [
    ('Logistic Regression', logistic_regression, logistic_params),
    ('Naive Bayes', naive_bayes, bayes_params),
    ('SVM', svm, svm_params)
]















### Train and Hypertune the models
# Define a function to train and hypertune all model types
def train_and_evaluate_classifiers(X, y, X_val, y_val, X_test, y_test, predefined_split):
    print("\nTraining models with hyperparameter tuning...\n")
    # Create an array to hold the result of training and hyperparameter tuning all models
    tuning_results = [["Model Name", "Best hyperparameters", "Validation accuracy of best model"]]

    # Array to hold the retrived best tuned model for each classifier type to use later in testing
    best_models = []

    for model_name, classifier, hyperparameters in classifiers:
        best_model, best_params, best_validation_acc = hypertune(X, y, X_val, y_val, classifier, hyperparameters, predefined_split)
        tuning_results.append([model_name, best_params, best_validation_acc])
        best_models.append((model_name, best_model))

    print("TUNING RESULTS")
    for row in tuning_results:
        print(row)


    ### Evaluate models on the testing set
    print("\nTesting models...\n")

    # Create an array to hold the result of testing accuracies on each classifer model
    testing_results = [["Model Name", "Testing Accuracy"]]
    for model_name, model in best_models:
        # Predict on the testing data
        y_pred = model.predict(X_test)
        # Calculate testing accuracy for the model
        testing_accuracy = accuracy_score(y_test, y_pred)
        testing_results.append([model_name, testing_accuracy])

    print("TESTING ACCURACIES")
    for row in testing_results:
        print(row)


# First evaluate and compare models on a preprocessed dataset that has been lemmatized only
print("\n\nTraining Models on dataset preprocessed with lemmatization only\n")
X, y, X_val, y_val, X_test, y_test, predefined_split = getProcessedSets(lemmatize = True, stem = False, remove_stopwords= False)
train_and_evaluate_classifiers(X, y, X_val, y_val, X_test, y_test, predefined_split)


# Then evaluate and compare models on a preprocessed dataset that has been stemmed only
print("\n\nTraining Models on dataset preprocessed with stemming only\n")
X, y, X_val, y_val, X_test, y_test, predefined_split = getProcessedSets(lemmatize = False, stem = True, remove_stopwords= False)
train_and_evaluate_classifiers(X, y, X_val, y_val, X_test, y_test, predefined_split)


# We then evaluate and compare models on a preprocessed dataset that has been lemmatized and in which we removed stop words as well
print("\n\nTraining Models on dataset preprocessed with lemmatization and stop words removed\n")
X, y, X_val, y_val, X_test, y_test, predefined_split = getProcessedSets(lemmatize = True, stem = False, remove_stopwords= True)
train_and_evaluate_classifiers(X, y, X_val, y_val, X_test, y_test, predefined_split)

# Finally, as a comparison basis with the previous pre processing techniques,
# compare models on a dataset that is only vectorized (so not lemmatized, stemmed and where stop words are not removed)
print("\n\nTraining Models on dataset not preprocessed\n")
X, y, X_val, y_val, X_test, y_test, predefined_split = getProcessedSets(lemmatize = False, stem = False, remove_stopwords= False)
train_and_evaluate_classifiers(X, y, X_val, y_val, X_test, y_test, predefined_split)

print("\n\n\n")


