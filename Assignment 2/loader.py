# IMPORTS

# For WSD and Lesk's algorithm
import xml.etree.cElementTree as ET # For XML parsing
import codecs # For encoding and decoding operations
import nltk # For natural language processing
from nltk.corpus import stopwords # For stop words removal
from nltk.stem import WordNetLemmatizer # For lemmatization
from nltk.tokenize import word_tokenize # For tokenization
from nltk.corpus import wordnet as wn # For WordNet
from nltk.wsd import lesk # For Lesk's algorithm

#  For bootstrapping
import matplotlib.pyplot as plt # For plotting
from collections import Counter # For counting the frequency of words in the test set
import pandas as pd # For loading the created dataset for seeding
from sklearn.feature_extraction.text import TfidfVectorizer # For vectorizing the dataset
from sklearn.linear_model import LogisticRegression # For training the model
import numpy as np # For manipulating arrays

# For Large Language Model
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ensure that NLTK resources are downloaded (only need to do this once)
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')



# Class to represent a WSD instance
class WSDInstance:
    def __init__(self, my_id, lemma, context, index):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index      # index of lemma within the context
    
    # String representation of WSDInstance
    def __str__(self):
        '''
        For printing purposes.
        '''
        return '%s\t%s\t%s\t%d' % (self.id, self.lemma, ' '.join(self.context), self.index)

def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    # parse XML file
    tree = ET.parse(f)
    # get root of XML file
    root = tree.getroot()
    
    dev_instances = {}
    test_instances = {}
    
    # iterate through all elements in the root
    for text in root:
        # check if the text is in the dev or test set
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            # construct sentence context
            # Creating a context list containing the lemmas of all words in the sentence.
            context = [to_ascii(el.attrib['lemma']) for el in sentence]
            
            # iterate through all the elements in the sentence
            for i, el in enumerate(sentence):
                # check if the element is an instance
                if el.tag == 'instance':
                    # get the id of the instance
                    my_id = el.attrib['id']
                    # get the lemma of the instance and convert it to ASCII
                    lemma = to_ascii(el.attrib['lemma'])
                    # Creating a WSDInstance and adding it to the respective dictionary.
                    instances[my_id] = WSDInstance(my_id, lemma, context, i)
    return dev_instances, test_instances


# Function to load the keys (correct answers) for the WSD instances.
def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys. 
    '''
    dev_key = {}
    test_key = {}

    # iterate through all lines in the file
    for line in open(f):
        # skip empty lines
        if len(line) <= 1: continue

        # Splitting the line into document ID, instance ID, and sense key.
        doc, my_id, sense_key = line.strip().split(' ', 2)
        
        # check if the key is in the dev or test set
        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key

def to_ascii(s):
    """
    Function to convert a string to ASCII.
    """
    # Removes all non-ascii characters.
    return codecs.encode(s, 'ascii', 'ignore').decode('ascii')


# Function to preprocess a given context
def preprocess_context(context, join_sentence = True):
    """
    Preprocess a given context by tokenizing, removing stop words, and lemmatizing.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Tokenization
    if join_sentence:
        tokens = word_tokenize(' '.join(context))
    else:
        tokens = word_tokenize(context)

    # Adjust the condition to include words with periods and hyphens
    def is_valid_token(token):
        return token.isalpha() or '.' in token or '-' in token
    
    # Stop Word Removal and Lemmatization
    processed_context = [
        lemmatizer.lemmatize(token.lower())
        for token in tokens
        if token.lower() not in stop_words and is_valid_token(token)
    ]
    
    return processed_context


# Function to preprocess the context of a WSDInstance by tokenizing, removing stop words, and lemmatizing.
def preprocess_instance(wsd_dataset):
    """
    Preprocess the context of a WSDInstance by tokenizing, removing stop words, and lemmatizing.
    """
    newContext = preprocess_context(wsd_dataset.context)
    
    # Update the context of the WSDInstance
    wsd_dataset.context = newContext
    return wsd_dataset


# Function to check the version of WordNet. Must be 3.0.
def check_wordnet_version():
    """
    Check the version of WordNet. Must be 3.0.
    """
    if wn.get_version() != '3.0':
        raise ValueError(f"Wordnet version is {wn.get_version()}. Must be 3.0.")




# Function to get the most frequent sense of the word from WordNet's synset.
# This serves as the baseline model in our experiments.
def get_most_frequent_sense(word):
    """
    Get the most frequent sense of the word from WordNet's synset.
    As specified on Ed: "the evaluation dataset is constructed assuming only nouns are to be disambiguated"
    """
    synsets = wn.synsets(word)
    return synsets[0].name() if synsets else None
    


# Function to apply Lesk's algorithm to find the sense of the ambiguous_word given the context_sentence.
def lesk_model(context_sentence, ambiguous_word):
    """
    Apply Lesk's algorithm to find the sense of the ambiguous_word given the context_sentence.
    """
    best_synset = lesk(context_sentence, ambiguous_word)
    return best_synset.name() if best_synset else None






# Function to parse a gold standard sense key into its components and return them
# Note that the group category number is ignored as it is not returned by either the baseline nor Lesk's algorithm 
def parse_gold_sense_key(sense_key):
    """
    Parse a sense key into its components.
    """
    # Split the sense key into its components
    parts = sense_key.split('%')

    # Extract the lemma, part of speech, and sense number, ignore the group catgeory number
    lemma = parts[0]
    sense_info = parts[1].split(':')
    pos = sense_info[0]
    sense_num = sense_info[2]

    # Return the lemma, part of speech, and sense number
    return lemma, pos, sense_num






# Function to compare a predicted sense from either the baseline or Lesk's algorithm to a list of gold standard sense keys.
# Since the gold standard can have mutliple sense keys, we need to check if the predicted sense is in that list of senses.
def compare_sense_keys(y_hat_mfs, y_gold, multiple_senses=True):
    """
    Compare a predicted sense (from NLTK synset) to a list of gold standard sense keys.
    """
    # Extract information from the predicted sense, get the lemma, part of speech, and sense number
    y_hat_lemma = y_hat_mfs.split('.')[0]
    y_hat_pos = y_hat_mfs.split('.')[1]
    y_hat_sense_num = y_hat_mfs.split('.')[2]

    # Map POS from synset to sense key format
    pos_map = {'n': '1', 'v': '2', 'a': '3', 's': '5', 'r': '4'}
    y_hat_pos_number = pos_map.get(y_hat_pos, '0')

    if multiple_senses:
        # Iterate over the gold standard senses and parse each sense key
        for sense_key in y_gold:
            gold_lemma, gold_pos, gold_sense_num = parse_gold_sense_key(sense_key)

            # Compare the lemma, part of speech, and sense number
            if y_hat_lemma == gold_lemma and y_hat_pos_number == gold_pos and int(y_hat_sense_num)-1 == int(gold_sense_num):
                return True
    else:
        # Parse the first gold standard sense key only
        gold_lemma, gold_pos, gold_sense_num = parse_gold_sense_key(y_gold[0])

        # Compare the lemma, part of speech, and sense number
        if y_hat_lemma == gold_lemma and y_hat_pos_number == gold_pos and int(y_hat_sense_num)-1 == int(gold_sense_num):
            return True

    # If no match is found in the gold standard list of possible senses, return false
    return False




# Function to compute the accuracy of the baseline and Lesk's algorithm and print them as a basis of comparison.
def compare_baseline_lesk(X, y, multiple_senses=True):
    """
    Compute the accuracy of WSD methods.
    """
    correct_mfs = 0
    correct_lesk = 0
    
    # Iterate through all instances
    for key, instance in X.items():
        # Get the gold standard sense
        y_gold = y.get(key)

        # Predict using the most frequent sense
        y_hat_mfs = get_most_frequent_sense(instance.lemma)
        # Check if the prediction is correct, add 1 to the corresponding correct counter if so.
        if compare_sense_keys(y_hat_mfs, y_gold, multiple_senses=multiple_senses):
            correct_mfs += 1            
        
        # Predict using Lesk
        y_hat_lesk = lesk_model(instance.context, instance.lemma)
        # Check if the prediction is correct, add 1 to the corresponding correct counter if so.
        if compare_sense_keys(y_hat_lesk, y_gold, multiple_senses=multiple_senses):
            correct_lesk += 1

        if key == 'd004.s012.t001':
            print("Sample predicted by Lesk's:", y_hat_lesk)
            print("Sample predicted by baseline:", y_hat_mfs)
            print("Actual:", y_gold)
            print()

    # Compute the accuracy of the two methods
    total_instances = len(y)

    # Accuracy = # of correct predictions / # of total instances
    accuracy_mfs = correct_mfs / total_instances
    accuracy_lesk = correct_lesk / total_instances
    
    print(f"Accuracy of Most Frequent Sense: {accuracy_mfs:.2f}")
    print(f"Accuracy of Lesk Algorithm: {accuracy_lesk:.2f}")














def plot_test_word_distribution(test_set, top_n):
    """
    Plot a histogram showing the distribution of the top N most frequent words in the test set.
    """
    # Count the frequency of each word in the test set
    word_counts = Counter(test_set)

    # Get the top N most common words
    most_common = word_counts.most_common(top_n)
    words, counts = zip(*most_common)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.title(f'Top {top_n} Most Frequent Words in Test Set')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.show()
    return words



def load_seed_dataset(file_path, word):
    """
    Load the seed dataset for a specific word.
    """
    # Load the entire merged dataset
    seed_dataset = pd.read_csv(file_path)

    # Filter the dataset for the specific word
    seed_specific_dataset = seed_dataset[seed_dataset['word'] == word]

    # Extract sentences (X) and labels (y)
    X_seed_word = seed_specific_dataset['sentence'].values
    y_seed_word = seed_specific_dataset['label'].values

    return X_seed_word, y_seed_word            


def bootstrap_model(X_seed, y_seed, X_test, max_iterations=15, confidence_threshold=0.6):
    """
    Train a model in an unsupervised matter. Use the bootstrapping method to iteratively train the model on a seed dataset and test it on the test dataset.
    """
    # Initialize TF-IDF Vectorizer 
    vectorizer = TfidfVectorizer()

    # Vectorize the seed dataset
    X_seed_vect = vectorizer.fit_transform(X_seed)
    X_seed_vect = X_seed_vect.toarray()

    # Vectorize the test dataset
    X_test_vect = vectorizer.transform(X_test)
    X_test_vect = X_test_vect.toarray()

    # Initialize the Logistic Regression model
    model = LogisticRegression()

    for _ in range(max_iterations):
        # Train the model
        model.fit(X_seed_vect, y_seed)

        # Make predictions and get probabilities to select most confident predictions and update the seed set accordingly
        probabilities = model.predict_proba(X_test_vect)
        predictions = model.predict(X_test_vect)

        # Select most confident predictions and update the seed set accordingly
        confident_indices = np.where(np.max(probabilities, axis=1) > confidence_threshold)[0]

        # Update the seed dataset and vectorize it
        X_seed_vect = np.concatenate((X_seed_vect, X_test_vect[confident_indices]))
        y_seed = np.concatenate((y_seed, predictions[confident_indices]))

        # Remove the confident predictions from the test dataset
        X_test_vect = np.delete(X_test_vect, confident_indices, axis=0)

        # Break if all test examples have been removed from the test dataset
        if len(X_test_vect) == 0:
            break

    return model, vectorizer



def train_and_test_bootstrap(most_used_words, X_test, y_test):
    """
    Train and test a model for each of the 6 most frequent words in the test set.
    """
    # Iterate through the 6 most frequent words in the test set and train a model for each of them.
    for word in most_used_words:
        print(f"Training model for {word}...")
        # Get the seed set corresppoinding to the word
        X_seed_word, y_seed_word = load_seed_dataset('seed_set.csv', word)

        # Prepocessing of the seed set
        for i in range(len(X_seed_word)):
            X_seed_word[i] = preprocess_context(X_seed_word[i], join_sentence=False)
        X_seed_word = [' '.join(context) for context in X_seed_word] # Revert tokenized context back to string
        
        # Get the test set examples corresponding to the word
        X_test_word = {k:v for (k,v) in X_test.items() if v.lemma == word}

        # Get the context matrix of the test dataset, use .join to make it a string and not a list of string.
        X_test_context = [instance.context for instance in X_test.values()]
        X_test_context = [' '.join(context) for context in X_test_context] # Revert tokenized context back to string

        # Train a model using the bootstrapping method
        model, vectorizer = bootstrap_model(X_seed_word, y_seed_word, X_test_context)  

        # EVAL MODE

        # Vectorize the test dataset
        X_test_vect = vectorizer.transform(X_test_word)
        X_test_vect = X_test_vect.toarray()

        # Make predictions on the test set and evaluate the model
        y_pred = model.predict(X_test_vect)
        
        # Get the gold standard labels corresponding to the word
        y_test_word = {k:v for (k,v) in y_test.items() if k in X_test_word}

        # Compute the accuracy of the model by using compare_sense_keys
        correct = 0
        i = 0
        for key, instance in X_test_word.items():
            if compare_sense_keys(y_pred[i], y_test_word[key]):
                correct += 1
            i += 1
        
        accuracy = correct / len(y_test_word)

        print(f"Accuracy of model for {word}: {accuracy:.2f}")
        print(f"Predicted: {y_pred}")
        print(f"Actual: {y_test_word.values()}")
        print()





def llm_compute_accuracy(pred):
    """
    Compute the accuracy of the llm model on both the validation and test sets.
    """
    # Get the predictions and labels
    labels = pred.label_ids
    # Get the predicted labels
    preds = pred.predictions.argmax(-1)
    # Calculate the accuracy
    acc = accuracy_score(labels, preds)
    # Return the accuracy
    return {'accuracy': acc}


def load_and_preprocess_dataset(file_path):
    """
    Load and preprocess the imported dataset.
    """
    # Read the dataset
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Extract the sentences and labels
    sentences = []
    labels = []

    for line in lines:
        # Split the line into sentence and label
        parts = line.strip().split('\t')
        # if the line has both sentence and label

        if len(parts) == 2:
            sentence, label = parts

            # Pre process the sentence and append it to the sentences list
            processed_sentence = ' '.join(preprocess_context(sentence, join_sentence=True))
            sentences.append(processed_sentence)

            # Append the label to the labels list
            labels.append(label)

    # Split the dataset into train, validation, and test sets - 80% train, 10% validation, and 10% test
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)
    val_sentences, test_sentences, val_labels, test_labels = train_test_split(test_sentences, test_labels, test_size=0.5, random_state=42)

    # Tokenization using BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_sentences, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_sentences, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_sentences, truncation=True, padding=True, max_length=128)

    # Return the encodings and labels
    return train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels, tokenizer


# Custom Dataset Class for training our Large Language Model
class WSDDataset(Dataset):
    # Constructor
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # Returns an item from the dataset at the given index
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx].split('.')[-1]))  # label format is word.pos.sense
        return item

    # Returns the length of the dataset
    def __len__(self):
        return len(self.labels)




if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    X_val, X_test = load_instances(data_f)
    y_val, y_test = load_key(key_f)

    # remove instances that do not have keys in both val and test sets
    X_val = {k:v for (k,v) in X_val.items() if k in y_val}
    X_test = {k:v for (k,v) in X_test.items() if k in y_test}
    
    # Check the length of the dictionaries as a sanity check.
    print("Before preprocessing:")
    print(f"Length of X_val: {len(X_val)}") # number of val instances
    print(f"Length of X_test: {len(X_test)}") # number of test instances
    # Print the an instance in the val set, as a sanity check.
    print(X_val['d001.s001.t002'])

    # Preprocess the data
    for key, instance in X_val.items():
        X_val[key] = preprocess_instance(instance)
    for key, instance in X_test.items():
        X_test[key] = preprocess_instance(instance)

    # Check the length of the dictionaries as a sanity check after preprocessing.
    print("After preprocessing:")
    print(f"Length of X_val: {len(X_val)}") # number of val instances
    print(f"Length of X_test: {len(X_test)}") # number of test instances
    # Print the an instance in the val set, as a sanity check  after preprocessing.
    print(X_val['d001.s001.t002'])
    print() 

    # Check the version of word net, as a sanity check. Should be 3.0.
    check_wordnet_version()

    # Compute the accuracy of the baseline and Lesk's algorithm and print them as a basis of comparison on the validation set
    print("Accuracies of baseline and Lesk's algorithm on validation set (without considering all senses in the gold labels):")
    compare_baseline_lesk(X_val, y_val, multiple_senses=False)
    print()
    print("Accuracies of baseline and Lesk's algorithm on validation set (considering all senses in the gold labels):")
    compare_baseline_lesk(X_val, y_val, multiple_senses=True)
    print()
    
    # Compute the accuracy of the two methods (the best result was with considering all senses in the gold labels)
    print("Accuracies of baseline and Lesk's algorithm on test set (considering all senses in the gold labels):")
    compare_baseline_lesk(X_test, y_test, multiple_senses=True)
    
    print("\n\n")

    # Plot the distribution of words in the test set (only top six words), return these top 6 most frequent words in an array.
    most_used_words = plot_test_word_distribution([instance.lemma for instance in X_test.values()], top_n=6)
    
    # Print the possible senses and their definitions of the most frequent words in the test set using WordNet's synset.
    # This is used to generate the seed set using a Large Language Model.
    for word in most_used_words:
        synsets = wn.synsets(word)
        print(f"Possible senses of {word}:")
        for synset in synsets:
            print(f"{synset.name()}: {synset.definition()}")
        print()

    # Train and test a model for each of the 6 most frequent words in the test set.
    train_and_test_bootstrap(most_used_words, X_test, y_test)


    print("\n\n") 

    # Check if GPU is available, if not use CPU
    if not torch.backends.mps.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    # Load and preprocess the dataset
    file_path = 'dataset-1mb.txt'  
    train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels, tokenizer = load_and_preprocess_dataset(file_path)
    train_dataset = WSDDataset(train_encodings, train_labels)
    val_dataset = WSDDataset(val_encodings, val_labels)
    test_dataset = WSDDataset(test_encodings, test_labels)

    # Load a pre-trained model (BERT)
    num_labels = len(set(train_labels))  # Number of unique labels
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels).to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',         # output directory for checkpoints and predictions
        num_train_epochs=1,             # total number of training epochs          
        per_device_train_batch_size=16, # batch size for training
        per_device_eval_batch_size=16,  # batch size for evaluation 
        logging_dir='./logs',           # directory for storing logs 
        logging_steps=10,               # number of steps between logging
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=llm_compute_accuracy
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the validation set
    print("Evaluating model on validation set...")
    val_results = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Accuracy on validation set: {val_results['eval_accuracy']:.2f}")

    # Evaluate the model on the test set
    print("Evaluating model on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Accuracy on test set: {test_results['eval_accuracy']:.2f}")
