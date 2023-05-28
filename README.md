# Sequence-to-Sequence Transliteration with Attention

The project fulfills the following:-

1. Learn to create sequence-to-sequence learning model problems using Recurrent Neural Networks (RNNs).
2. Compare the performance of model with different RNN cell types, including vanilla RNN, LSTM, and GRU.
3. comparsion of attention network with vanilla seq2seq model
4. Visualize the interactions between different components in an RNN-based model.

## Files:
Transliteration_model_without_attention.py  -- .py file to be run in terminal for model without attention and is directly integrated to wandb.PLease copy and paste the link of path_test,path_train and path_valid in the __main__ function whie running the code.
attention_mech.py  -- .py file to be run in terminal for model with attention and is directly integrated to wandb.PLease copy and paste the link of path_test,path_train and path_valid in the __main__ function whie running the code.  

### Arguments supported:-    
--wandb_project,default="CS6910-ASSIGNMENT_3"  
--learning_rate, type=float, default =0.001  
--hidden_dim, type=float, default=256  
--layer_dim, type=int, default=2  
--weight_decay,type=float, default=0.0001  
--embed_dim, default=256  
--epochs, default=10  
--optimizer, default='Adam'  
--batch_size,type=int,default=64  
--cell, type=str, default='LSTM'  
--drop_out, type=float, default=0.2  
--bi_dir, type=bool, default=True    
To run these files, go to the termianal, source folder and give python file_name.py --arguments  
**CS6910_ASSIGNMENT_3_without_attention_sweeps.ipynb** - Contains the complete code of model without attention and partial sweep results.  
**CS6910_ASSIGNMENT_3-BEAM SEARCH.ipynb**    - Contais model without attentiuon ad sweep results.  
**test_predictions_sample.csv** - contains the prediction vs true test data in .csv format
**best_accuracy_model_sample.ipynb** - contains source code of the above predictions
**test_predictions_attention.csv** - contains the prediction vs true data  of attention model in.csv format  
**cs6910_ASSIGNMENT_3=ATTENTION-MODEL-ATTENTION HEATMAP.ipynb** -source code of above predictions  
**cs6910 -assignment_3-attention-model1.ipynb** - contains a different approach for attention model and some sample sweeps  


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
The data is given to the encoder as the indices of the characters. We add the start and end token and pad the words accordingly to the maximum length. If there is any unknown character in the test or validation, its replaced by unknown character as well.

### Decoder

The decoder in the seq2seq model takes the context vector produced by the encoder and generates the output sequence. It also uses an RNN cell, which can be the same type as the encoder's RNN cell or a different one. At each time step, the decoder uses the current input character and the previous hidden state to predict the next character in the output sequence. The decoding process continues until an end-of-sequence token is generated or a maximum length is reached.  
Data can be given to the decoder in two forms during training. If we give the true/groundtruth/target as the input at each timestep, it is called as teacher forcing. Otherwise, we give the prediction as the next input and this is repeated during validation and testing.  

### Attention Mechanism

While the basic seq2seq model with an encoder and decoder can produce reasonable results, it has limitations in handling long input sequences and capturing relevant information. Attention mechanisms address these limitations by allowing the decoder to focus on different parts of the input sequence while generating the output sequence.    
As per the observation, attention mechanism improved the character level accuracy by very high vallue, owing to giving attention to specific characters while predicting similar meaning characters.  

The attention mechanism calculates attention weights for each input sequence element based on its relevance to the current decoding step. These attention weights are used to weight the contribution of each input element to the context vector calculation. By giving higher importance to relevant parts of the input sequence, the attention mechanism improves the model's ability to generate accurate outputs.

### Without Attention

In addition to the attention-based seq2seq model, this project also explores a variant without attention. The without attention model relies solely on the final hidden state of the encoder as the context vector for generating the output sequence. This model serves as a baseline for comparison against the attention-based models.

### Best hyperparameter configuration
learning_rate = 0.001    
epochs = 20    
cell = 'LSTM'    
bi_dir = True    
embed_dim=128    
hidden_dim = 256    
optimizer = 'NAdam'    
teacher_forcing_ratio = 0.5    
weight_decay = 0.001      
drop_out = 0.2    
validation_word_accuracy = 39.014%    
validation__char__accuracy = 88.892%    
Test _word_accuracy = 35%    

## Result
**LSTM** was found to be the best model compared to GRU, and RNN  
The model worked well in presence of bidirection and more layers compared to single layers  
Highest character level accuracy ws found for attention model whereas word accuracy were comparable  
Beam search also increased the character accuracy even though word accuracy ws simlar  
Attention model took more time to run compare to other models  

## Reference
- Pytorch Documentation  
- CS6910 -NPTEL
- CHATGPT FOR SYNTAX CORRECTIONS AND BASIC THEORIES  
- PYTORCH NEURAL MACHINBE SEQ TO SEQ TRANSLATION  
- kaggle.com
- towardsdatascience.com  
- padh.ai  
