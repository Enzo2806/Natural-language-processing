# Assignment 2: Word Sense Disambiguation Project

## Goal of the Assignment
This project focuses on exploring various Word Sense Disambiguation (WSD) techniques, specifically targeting the SemEval 2013 Shared Task #12 dataset and the FEWS dataset. We implemented several methodologies including a baseline approach, Lesk’s algorithm, a bootstrapping method using Yarowsky’s algorithm, and a fully supervised learning approach utilizing the BERT model. Our objective was to analyze and compare the effectiveness of these methods in accurately disambiguating word senses.

## Methodology
- **Baseline and Lesk’s Algorithm**: Implemented using WordNet and NLTK, focusing on the most frequent sense and context overlap.
- **Bootstrapping Method**: Employed Yarowsky’s algorithm to disambiguate specific high-frequency words, creating seed sets with the aid of ChatGPT 3.5.
- **Supervised Learning**: Utilized a BERT model fine-tuned on the FEWS dataset, which provides extensive labeled examples for WSD.

## Running the Project
To run this project, ensure Python 3 is installed along with the necessary libraries. Install the required Python packages using the following command:
```bash
pip install -r requirements.txt
```

## Conclusion

The project demonstrated various degrees of success with each methodology:
- The baseline method achieved a surprisingly high accuracy, suggesting that the most frequent sense might be a stronger baseline than anticipated.
- Lesk’s algorithm underperformed, likely due to its reliance on dictionary definitions which may not capture real-world usage accurately.
- The bootstrapping method showed polarized results, being highly effective in some cases and less so in others, depending on the diversity of word senses.
- The supervised learning approach, while not directly comparable to the other models, showed promising results and highlighted the potential of fine-tuning models on large, labeled datasets.
These findings provide valuable insights into the challenges and potential strategies for improving WSD systems.

## Report

The report of this assignment can be found below.

![A2_Report_Page_1](https://github.com/user-attachments/assets/16335a5f-e0b1-4de3-8ace-ba4cc24c65b8)
![A2_Report_Page_2](https://github.com/user-attachments/assets/9a1513d8-08ac-47b1-bb58-f46b1b0fbc57)
