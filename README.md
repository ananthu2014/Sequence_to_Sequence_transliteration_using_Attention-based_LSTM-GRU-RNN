# Sequence-to-Sequence Transliteration with Attention

This project aims to address the goals mentioned below:

1. Learn to create sequence-to-sequence learning model problems using Recurrent Neural Networks (RNNs).
2. Compare the performance of model with different RNN cell types, including vanilla RNN, LSTM, and GRU.
3. comparsion of attention network with vanilla seq2seq model
4. Visualize the interactions between different components in an RNN-based model.


## Overview

Sequence-to-sequence (seq2seq) learning is a powerful technique used in various natural language processing tasks, such as machine translation and transliteration. In this project, we focus on the transliteration task from the dataset sample of [aksharantar](https://drive.google.com/file/d/1uRKU4as2NlS9i8sdLRS1e326vQRdhvfw/view) released by [AI4Bharath](https://ai4bharat.org/) which involves transliteration of one language to another.

## Training model

### Apporach
To solve this problem, you will need to train a sequence-to-sequence model that can learn the mapping between the Latin script and the Devanagari script. You can use various recurrent neural network (RNN) architectures, such as vanilla RNN, LSTM, or GRU, to build your model. These models are effective in capturing sequential information and can be trained using the provided dataset.

### Packages
The following dependencies are required to run the project:

- Python (version X.X.X)
- pytorch (version X.X.X)
- NumPy (version X.X.X)
- Matplotlib (version X.X.X)
- pandas
- seaborn
- wandb

### Encoder

The encoder in the seq2seq model is responsible for processing the input sequence and producing the context vector. It typically consists of a recurrent neural network (RNN) cell, such as a vanilla RNN, LSTM, or GRU. The encoder reads the input sequence one character at a time and updates its internal state at each time step. The final state of the encoder represents the context vector, which encodes the input sequence's information.

### Decoder

The decoder in the seq2seq model takes the context vector produced by the encoder and generates the output sequence. It also uses an RNN cell, which can be the same type as the encoder's RNN cell or a different one. At each time step, the decoder uses the current input character and the previous hidden state to predict the next character in the output sequence. The decoding process continues until an end-of-sequence token is generated or a maximum length is reached.

### Attention Mechanism

While the basic seq2seq model with an encoder and decoder can produce reasonable results, it has limitations in handling long input sequences and capturing relevant information. Attention mechanisms address these limitations by allowing the decoder to focus on different parts of the input sequence while generating the output sequence.

The attention mechanism calculates attention weights for each input sequence element based on its relevance to the current decoding step. These attention weights are used to weight the contribution of each input element to the context vector calculation. By giving higher importance to relevant parts of the input sequence, the attention mechanism improves the model's ability to generate accurate outputs.

### Without Attention

In addition to the attention-based seq2seq model, this project also explores a variant without attention. The without attention model relies solely on the final hidden state of the encoder as the context vector for generating the output sequence. This model serves as a baseline for comparison against the attention-based models.

### Best hyperparameter configuration

## Result

## Reference
- 
