# Natural Language Processing Assignments

## Project Overview
This repository contains two assignments completed for a Natural Language Processing course. These assignments explore different aspects of machine learning and natural language processing techniques, specifically focusing on text classification and word sense disambiguation.

## Assignments

### Assignment 1: Animal Facts Classification
**Objective**: To classify animal facts as real or fake using various linear classifiers and preprocessing techniques. This assignment examines the capabilities and limits of classifiers like SVM, Logistic Regression, and Naïve Bayes in a controlled experimental setting.

**Key Components**:
- **Dataset**: Generated using ChatGPT 3.5, consisting of 100 real and 100 fake facts about different animals.
- **Methodology**: Experimentation with different preprocessing techniques and classifiers.
- **Results**: SVM proved to be the most effective classifier, with preprocessing variations showing minimal impact on the outcomes.

**[Read More About Assignment 1](/Assignment_1/README.md)**

### Assignment 2: Word Sense Disambiguation
**Objective**: To apply and compare various Word Sense Disambiguation (WSD) techniques on the SemEval 2013 Shared Task #12 dataset and the FEWS dataset, using methods ranging from traditional algorithms to modern deep learning approaches.

**Key Components**:
- **Methodologies**: Baseline frequency method, Lesk’s algorithm, bootstrapping with Yarowsky’s algorithm, and supervised learning with BERT.
- **Techniques**: Implementation included the use of NLTK’s WordNet interface, custom bootstrapping, and fine-tuning a pre-trained BERT model.
- **Results**: Varied success across methods, highlighting the strengths and limitations of each approach in WSD.

**[Read More About Assignment 2](/Assignment_2/README.md)**

## Running the Projects
Each assignment directory contains a `requirements.txt` file for setting up the necessary environment. Use the following command in the respective assignment directory to install dependencies:
```bash
pip install -r requirements.txt
```
