# BERT + GPT + VIT - Combination of Transformers

## Objective

A single modularised transformer file to call the 3 models GPT + VIT + BERT 

## Files Handled

### config.py

config.py that defines three different configuration functions for three different machine learning models: BERT, GPT, and ViT. These functions return dictionaries with various configuration parameters for each model.

### model.py

Contains the implementation of several transformer-based models: BERT, GPT, and ViT. These models are used for various natural language processing (NLP) and computer vision tasks. Here is a breakdown of the code and its functionalities:

#### BERT, GPT, and ViT Models
This module provides implementations of BERT (Bidirectional Encoder Representations from Transformers), GPT (Generative Pre-trained Transformer), and ViT (Vision Transformer) models, all based on the transformer architecture. Each of these models is defined as a class within the module.

##### BERT
The BERT model is designed for bidirectional language understanding and is suitable for various NLP tasks.

##### GPT
The GPT model is a generative model that predicts the next token in a sequence, making it suitable for text generation tasks.

##### ViT
The ViT model is designed for computer vision tasks and uses transformers to process image data.

#### Transformer Architecture Blocks
Within each model, the following transformer architecture components are implemented:

##### Attention Mechanism: 
The attention mechanism is a crucial part of transformers. It computes attention scores between input elements and is used to capture dependencies between tokens.

##### Multi-Head Attention: 
Multi-head attention is used in both BERT and GPT models. It allows the model to focus on different parts of the input sequence.

##### Feedforward Neural Network (MLP): 
Transformers typically include feedforward neural networks that process the output of attention layers.

##### Positional Embeddings: 
Positional embeddings are added to the input sequence to provide information about the position of tokens in the sequence.

##### Layer Normalization:
Layer normalization is applied before and after certain components to stabilize training.

##### Residual Connections: 
Residual connections are used to propagate gradients effectively during training.


### train.py

#### BERT Training:

It defines a function get_bert_batch for getting batches of data.
Sets up the BERT model for training.
Trains the BERT model for a specified number of iterations, printing loss and weight updates.

#### GPT Training:

Defines functions for getting GPT data batches and estimating loss.
Sets up the GPT model for training.
Trains the GPT model for a specified number of iterations, printing loss on both training and validation data.

#### ViT Training:

Defines functions for training and testing steps for a Vision Transformer (ViT) model.
Trains the ViT model for a specified number of epochs, printing training and testing loss and accuracy metrics.


### transformers.py

Function called transformers_combined(name, iterations=10) which allows you to train different transformer-based models: BERT, GPT, and VIT.

## Training Log
The below shows some of the last few lines of the training log for each model

### BERT 
```
ITERATION: 475  | Loss 6.32  | ΔW: 0.4
ITERATION: 476  | Loss 6.29  | ΔW: 0.418
ITERATION: 477  | Loss 6.38  | ΔW: 0.377
ITERATION: 478  | Loss 6.33  | ΔW: 0.405
ITERATION: 479  | Loss 6.32  | ΔW: 0.399
ITERATION: 480  | Loss 6.31  | ΔW: 0.394
ITERATION: 481  | Loss 6.34  | ΔW: 0.406
ITERATION: 482  | Loss 6.39  | ΔW: 0.411
ITERATION: 483  | Loss 6.31  | ΔW: 0.413
ITERATION: 484  | Loss 6.32  | ΔW: 0.435
ITERATION: 485  | Loss 6.33  | ΔW: 0.414
ITERATION: 486  | Loss 6.31  | ΔW: 0.414
ITERATION: 487  | Loss 6.3  | ΔW: 0.413
ITERATION: 488  | Loss 6.34  | ΔW: 0.436
ITERATION: 489  | Loss 6.28  | ΔW: 0.425
ITERATION: 490  | Loss 6.33  | ΔW: 0.427
ITERATION: 491  | Loss 6.36  | ΔW: 0.437
ITERATION: 492  | Loss 6.41  | ΔW: 0.449
ITERATION: 493  | Loss 6.28  | ΔW: 0.475
ITERATION: 494  | Loss 6.32  | ΔW: 0.436
ITERATION: 495  | Loss 6.29  | ΔW: 0.449
ITERATION: 496  | Loss 6.37  | ΔW: 0.445
ITERATION: 497  | Loss 6.39  | ΔW: 0.444
ITERATION: 498  | Loss 6.38  | ΔW: 0.454
ITERATION: 499  | Loss 6.36  | ΔW: 0.434
ITERATION: 500  | Loss 6.27  | ΔW: 0.455
```

### GPT

```
Iteration:        230 | Train Loss 2.7008 | Validation Loss 6.4825
Iteration:        231 | Train Loss 2.6480 | Validation Loss 6.3979
Iteration:        232 | Train Loss 2.7001 | Validation Loss 6.6115
Iteration:        233 | Train Loss 2.6364 | Validation Loss 6.4800
Iteration:        234 | Train Loss 2.6516 | Validation Loss 6.5324
Iteration:        235 | Train Loss 2.6101 | Validation Loss 6.5856
Iteration:        236 | Train Loss 2.5883 | Validation Loss 6.5621
Iteration:        237 | Train Loss 2.5884 | Validation Loss 6.3255
Iteration:        238 | Train Loss 2.6290 | Validation Loss 6.5534
Iteration:        239 | Train Loss 2.6220 | Validation Loss 6.6908
Iteration:        240 | Train Loss 2.6259 | Validation Loss 6.5190
Iteration:        241 | Train Loss 2.6332 | Validation Loss 6.4944
Iteration:        242 | Train Loss 2.5699 | Validation Loss 6.5831
Iteration:        243 | Train Loss 2.5515 | Validation Loss 6.5357
Iteration:        244 | Train Loss 2.5824 | Validation Loss 6.6013
Iteration:        245 | Train Loss 2.5224 | Validation Loss 6.4881
Iteration:        246 | Train Loss 2.5619 | Validation Loss 6.6376
Iteration:        247 | Train Loss 2.5360 | Validation Loss 6.4721
Iteration:        248 | Train Loss 2.4306 | Validation Loss 6.5773
Iteration:        249 | Train Loss 2.5353 | Validation Loss 6.5059
Iteration:        250 | Train Loss 2.4792 | Validation Loss 6.5251
```


### ViT
```
Epoch: 25 | train_loss: 1.1304 | train_acc: 0.3047 | test_loss: 1.3959 | test_acc: 0.1979
Epoch: 26 | train_loss: 1.1160 | train_acc: 0.3945 | test_loss: 1.1323 | test_acc: 0.2604
Epoch: 27 | train_loss: 1.1474 | train_acc: 0.3047 | test_loss: 1.1028 | test_acc: 0.2604
Epoch: 28 | train_loss: 1.1495 | train_acc: 0.2969 | test_loss: 1.1626 | test_acc: 0.1979
Epoch: 29 | train_loss: 1.0972 | train_acc: 0.4258 | test_loss: 1.0218 | test_acc: 0.5417
Epoch: 30 | train_loss: 1.1661 | train_acc: 0.2812 | test_loss: 1.0615 | test_acc: 0.5417
Epoch: 31 | train_loss: 1.1245 | train_acc: 0.2930 | test_loss: 1.3248 | test_acc: 0.2604
Epoch: 32 | train_loss: 1.1022 | train_acc: 0.4336 | test_loss: 1.2106 | test_acc: 0.1979
Epoch: 33 | train_loss: 1.1486 | train_acc: 0.2930 | test_loss: 1.1365 | test_acc: 0.1979
Epoch: 34 | train_loss: 1.1337 | train_acc: 0.2773 | test_loss: 1.0962 | test_acc: 0.2604
Epoch: 35 | train_loss: 1.1472 | train_acc: 0.3164 | test_loss: 1.0449 | test_acc: 0.5417
Epoch: 36 | train_loss: 1.1387 | train_acc: 0.2930 | test_loss: 1.1721 | test_acc: 0.1979
Epoch: 37 | train_loss: 1.1098 | train_acc: 0.3008 | test_loss: 1.0785 | test_acc: 0.5417
Epoch: 38 | train_loss: 1.0921 | train_acc: 0.4219 | test_loss: 1.1427 | test_acc: 0.1979
Epoch: 39 | train_loss: 1.1070 | train_acc: 0.2930 | test_loss: 1.2060 | test_acc: 0.1979
Epoch: 40 | train_loss: 1.1235 | train_acc: 0.2930 | test_loss: 1.1702 | test_acc: 0.2604
Epoch: 41 | train_loss: 1.0915 | train_acc: 0.4375 | test_loss: 1.1651 | test_acc: 0.1979
Epoch: 42 | train_loss: 1.1275 | train_acc: 0.2930 | test_loss: 1.1333 | test_acc: 0.1979
Epoch: 43 | train_loss: 1.1337 | train_acc: 0.2969 | test_loss: 1.0577 | test_acc: 0.5417
Epoch: 44 | train_loss: 1.0832 | train_acc: 0.4062 | test_loss: 1.1199 | test_acc: 0.2604
Epoch: 45 | train_loss: 1.1400 | train_acc: 0.3047 | test_loss: 1.1687 | test_acc: 0.2604
Epoch: 46 | train_loss: 1.1354 | train_acc: 0.3047 | test_loss: 1.1585 | test_acc: 0.1979
Epoch: 47 | train_loss: 1.1113 | train_acc: 0.2969 | test_loss: 1.0857 | test_acc: 0.5417
Epoch: 48 | train_loss: 1.1107 | train_acc: 0.2773 | test_loss: 1.1000 | test_acc: 0.2604
Epoch: 49 | train_loss: 1.0927 | train_acc: 0.3906 | test_loss: 1.1426 | test_acc: 0.1979
Epoch: 50 | train_loss: 1.1131 | train_acc: 0.2930 | test_loss: 1.1673 | test_acc: 0.1979
```