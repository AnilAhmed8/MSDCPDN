# A Semi-Supervised Multi-View Siamese Network with Dual-Contextual Attention and Knowledge Distillation for Cross-Lingual Low-Resource Paraphrase Detection

This project implements a Siamese network for binary paraphrase detection using a multi-view knowledge distillation approach. The model is designed for training on an Arabic corpus and testing on an Urdu corpus, and supports both supervised (with ground-truth labels) and semi-supervised (using teacher pseudo-labels) training settings.

## Overview

In this project, we employ a Siamese network architecture that uses dual contextual encoders (BiLSTM and BiGRU) combined with word-level and sentence-level attention mechanisms. The outputs from both views are fused and then compared to compute a similarity score, which is used to determine if two texts are paraphrases.

The training procedure leverages knowledge distillation where a pretrained teacher model (whose parameters are frozen) guides the training of a student model via a combined loss function. The loss is composed of a KL divergence term (for matching softened outputs between teacher and student) and a Binary Cross-Entropy (BCE) loss for classification.

## Folder Structure
project/
├── dataset.py           # Defines the dataset class for preprocessed token sequences.
├── model.py             # Contains the Siamese model architecture definition.
├── preprocessing.py     # Handles text cleaning, tokenization, vocabulary building, and conversion to token indices.
├── train.py             # Contains training and evaluation routines with knowledge distillation.
├── utils.py             # Utility functions for computing evaluation metrics.
├── visualization.py     # Visualization routines for plotting confusion matrices and evaluation metrics.
└── main.py              # The main script that ties all modules together: data processing, model training, evaluation, and visualization.

## Features

- Siamese Network Architecture:  
  Dual-branch architecture with BiLSTM and BiGRU for capturing contextual information.

- Multi-View Attention:  
  Implements both word-level and sentence-level attention mechanisms to capture fine-grained and global information.

- Knowledge Distillation:
  Uses a pretrained teacher model to guide the student model training via a combined loss function (KL divergence and BCE loss).

- Preprocessing:  
  Includes text cleaning, tokenization, and vocabulary building tailored for Arabic and Urdu text.

- Evaluation and Visualization:  
  Computes key evaluation metrics (confusion matrix, precision, recall, F1 score, accuracy) and visualizes the results using Matplotlib.

## Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [Pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)

You can install the required packages using pip: pip install torch pandas scikit-learn matplotlib

Installation
	1.	Clone the repository:
        git clone <repository-url> 
        cd project

	2.	Ensure you have installed the required dependencies (see Requirements section).

Data Preparation
	•	Training Data (Arabic):
Prepare a CSV file named arabic_train.csv with the following columns:
	•	text1: First text in the pair.
	•	text2: Second text in the pair.
	•	label: Binary label (0 or 1) indicating if the texts are paraphrases.
	•	Test Data (Urdu):
Prepare a CSV file named urdu_test.csv with the same columns as above.

The preprocessing module in preprocessing.py cleans and tokenizes the text, builds a vocabulary (from Arabic data), and converts text into sequences of token indices.

Usage

The main script, main.py, ties everything together. It performs the following tasks:
	1.	Preprocessing:
Loads and cleans the Arabic training data and the Urdu test data. The vocabulary is built from the Arabic data and reused for Urdu.
	2.	Dataset Creation:
Creates dataset objects from preprocessed data that convert tokens into padded sequences.
	3.	Model Initialization:
Instantiates both the teacher and student models. The teacher model’s parameters are frozen to simulate a pretrained model.
	4.	Training:
Trains the student model using knowledge distillation. You can switch between supervised and semi-supervised training modes using a flag.
	5.	Evaluation:
Evaluates the trained student model on the Urdu test set by computing evaluation metrics.
	6.	Visualization:
Plots the confusion matrix and evaluation metrics (precision, recall, F1 score, and accuracy).

Running the Project

To run the project, execute the main script: python main.py

License

MIT License

Acknowledgments
	•	This project builds upon standard deep learning and knowledge distillation techniques.
	•	Special thanks to the contributors of PyTorch, scikit-learn, and Matplotlib for their valuable tools and libraries.
 
