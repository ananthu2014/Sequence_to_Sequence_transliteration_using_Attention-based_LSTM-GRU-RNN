#!/usr/bin/env python
# coding: utf-8

# In[85]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import random
import wandb
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import time
import math
import argparse


# In[86]:


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
device


# In[87]:


'''function to load the data'''
def load_data(path,language_names):
    df=pd.read_csv(path,header=None)
    df.columns=language_names
    return df


# In[88]:


'''Function for acquiring all the characters of the given data'''
def split_words(x):
    x=np.array(x)
    alpha=['_','\t','\n',' '] #pad token, start of word, end of word and unknown tokens
    b=[]
    for i in range(x.shape[0]):
        a=list(x[i])
        for j in range(len(a)):
            if a[j] not in b:
                b.append(a[j])
    b=sorted(b)
    alpha=alpha+b
    return alpha


# In[89]:


'''Functions to create the vocabulary dictionaries with their indices'''
def int_to_char(vocab):
    int2char={} #padding token, start of word, end of word token and unknown token
    for i in range(len(vocab)):
        int2char[i]=vocab[i][0]
    return int2char


# In[90]:


def char_to_int(int2char):
    char2int={ch:ii for ii,ch in int2char.items()}
    return char2int


# In[91]:


def process_data(df,english_vocab,hindi_vocab,
                 length_eng_max,length_hin_max,char2int_eng
                 ,char2int_hin):
    
    '''removing words of length more than max length'''
    df['English'] = df['English'].str.lower()
    df['transliteration_in_hindi'] = df['transliteration_in_hindi'].str.lower()
    df = df[df['English'].apply(len) <= length_eng_max-2]
    df = df[df['transliteration_in_hindi'].apply(len) <= length_hin_max-2]
    '''Adding start and end of word tokens'''
    y_og = df['transliteration_in_hindi'].values
    x_og = df['English'].values
    x = '\t'+x_og+'\n'
    y = '\t'+y_og+'\n'
    y_do=y_og+'\n'
    unknown=3
    pad=0
    pad_char='_'
    unknown_char=' '
    start=1
    end=2
    
    enc_input_data=torch.zeros(len(x),length_eng_max)
    dec_input_data=torch.zeros(len(y),length_hin_max)
    dec_output_data=torch.zeros(len(y),length_hin_max)
    for i, (xx,yy) in enumerate(zip(x,y)):
        for j,char in enumerate(xx):
            enc_input_data[i,j]=char2int_eng[char]
        #pad character is zero so no need of assigning it again
        for j,char in enumerate(yy):
            if char in hindi_vocab:
                dec_input_data[i,j]=char2int_hin[char]
            else:
                dec_input_data[i,j]=char2int_hin[unknown_char]
    
    for i, (xx,yy) in enumerate(zip(x,y_do)):
        for j,char in enumerate(yy):
            if char in hindi_vocab:
                dec_output_data[i,j]=char2int_hin[char]
            else:
                dec_input_data[i,j]=char2int_hin[unknown_char]
                
    return enc_input_data,dec_input_data,dec_output_data


# In[97]:


def one_hot_encoding(df,english_vocab,hindi_vocab,
                 length_eng_max,length_hin_max,char2int_eng,char2int_hin):
    
    
    '''removing words of length more than max length'''
    df = df[df['English'].apply(len) <= length_eng_max-2]
    df = df[df['transliteration_in_hindi'].apply(len) <= length_hin_max-2]
    '''Adding start and end of word tokens'''
    y = df['transliteration_in_hindi'].values
    x= df['English'].values
    x = '\t'+x+'\n'
    y = '\t'+y+'\n'
    
    unknown=3
    pad=0
    pad_char='_'
    unknown_char=' '
    start=1
    end=2
    
    encoder_input_data = np.zeros(
    (len(df['English']), length_eng_max, num_english_tokens), dtype="float32")
    decoder_input_data = np.zeros(
    (len(df['transliteration_in_hindi']), length_hin_max, num_hindi_tokens), dtype="float32")
    decoder_output_data = np.zeros(
    (len(df['transliteration_in_hindi']), length_hin_max, num_hindi_tokens), dtype="float32")
    pad_char='_'
    for i , (input_text,target_text) in enumerate(zip(x,y)):
        for t,char in enumerate(input_text):
            encoder_input_data[i,t,char2int_eng[char]]=1
        encoder_input_data[i,t+1:,char2int_eng[pad_char]]=1
    
        for t,char in enumerate(target_text):
            if char in hindi_vocab:
                decoder_input_data[i,t,char2int_hin[char]]=1
            else:
                decoder_input_data[i,t,char2int_hin[unknown_char]]=1
        decoder_input_data[i,t+1:,char2int_hin[pad_char]]=1
    
        '''decoder target data is one step ahead of decoder input data by one timestep
        and doesnot includes start token'''
        for t,char in enumerate(target_text):
            if t>0:
                if char in hindi_vocab:
                    decoder_output_data[i,t-1,char2int_hin[char]]=1
                else:
                    decoder_output_data[i,t-1,char2int_hin[unknown_char]]=1
                
        decoder_output_data[i,t:,char2int_hin[pad_char]]=1
    
    return torch.tensor(encoder_input_data),torch.tensor(decoder_input_data),torch.tensor(decoder_output_data)
    
    


# In[99]:





# In[50]:


def calculate_word_accuracy(dec_predicted_data, dec_output_data):
    # Here, we have to pass the arguments in the shape (batch_size, sequence_length,classes)
    dec_predicted_data = torch.argmax(dec_predicted_data, dim=-1)
    dec_output_data = torch.argmax(dec_output_data, dim=-1)
    
    with torch.no_grad():
        match = (dec_predicted_data == dec_output_data).all(dim=1)
        true_words = match.sum().item()
        batch_size = dec_predicted_data.shape[0]
    
    accuracy = (true_words / batch_size) * 100
    return accuracy


# In[51]:


def calculate_char_accuracy(decoder_predicted_data, decoder_output_data):
    #Here, we have to pass the arguments in the shape(batch_size,sequence_length,unique_tokens)
    batch_size, seq_length,unique_tokens = decoder_predicted_data.shape
    dec_predicted_data=torch.argmax(decoder_predicted_data,dim=-1)
    dec_output_data=torch.argmax(decoder_output_data,dim=-1)
    
    with torch.no_grad():
        correct_count = (dec_predicted_data == dec_output_data).sum().item()
        return (correct_count / (seq_length * batch_size))*100


# In[52]:


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# In[53]:


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[54]:


def data_loader(x,y,z,batch_size,device=device):
    
    x=x.to(device)
    y=y.to(device)
    z=z.to(device)
    combined=TensorDataset(x,y,z)
    loader=DataLoader(combined,batch_size=batch_size,shuffle=False,drop_last=True)#required in test data
    return loader


# In[100]:


class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, layer_dim, drop_out, bi_dir, cell,unique_tokens_eng):
        '''unique_token_hin is the third dimension in one hot encoding or no of tokens in eng or input size'''
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.cell = cell
        self.bi_dir = bi_dir
        self.unique_tokens_eng=unique_tokens_eng
        self.embed_dim=embed_dim
        self.drop_out=drop_out
        '''The input to the encoder will be of shape (batch_size,sequence_length) and the output size will be
        (batch_size,seq_length,embed_size)'''
        self.drop__out=nn.Dropout(p=self.drop_out)

        self.embedding = nn.Embedding(unique_tokens_eng, embed_dim) 
        self.relu=nn.ReLU()
        self.rnn=nn.RNN(embed_dim,hidden_dim,dropout=self.drop_out,
                        num_layers=layer_dim,bidirectional=bi_dir,batch_first=True)
        self.gru=nn.GRU(embed_dim,hidden_dim,dropout=self.drop_out,
                        num_layers=layer_dim,bidirectional=bi_dir,batch_first=True)
        self.lstm=nn.LSTM(embed_dim,hidden_dim,dropout=self.drop_out,
                         num_layers=layer_dim,bidirectional=bi_dir,batch_first=True)
        self.linear =nn.Linear(self.hidden_dim*(1+int(self.bi_dir)),self.hidden_dim)

    def forward(self, x,hidden):
        '''Here, x is the encoder input data'''
        batch_size=x.size(0)
        embedding_input = self.embedding(x)
        #embedding_input = self.drop__out(embedding_input)
        embedding_input = self.relu(embedding_input)
        embedding_input=embedding_input.to(device)
        if self.cell=="GRU":
            output,h_n=self.gru(embedding_input,hidden)
            return output,h_n
        elif self.cell=='RNN':
            output,h_n=self.gru(embedding_input,hidden)
            return output,h_n
        elif self.cell=='LSTM':
            output,(h_n,c_n)=self.lstm(embedding_input,hidden)
            return output,(h_n,c_n)
        
    def encoder_initial(self,batch_size,device=device):
        if self.cell == "LSTM":
            h_0 =torch.randn((1 + int(self.bi_dir)) * self.layer_dim, batch_size, self.hidden_dim, device=device)
            c_0 =torch.randn((1 + int(self.bi_dir)) * self.layer_dim, batch_size, self.hidden_dim, device=device) 
            return (h_0,c_0)
        #H_0,C_0 HAVE SAME DIMENSION
        else:
            h_0=torch.randn((1 + int(self.bi_dir)) * self.layer_dim, batch_size, self.hidden_dim, device=device)
            return h_0


# In[56]:


class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, layer_dim, drop_out, bi_dir, cell,unique_tokens_hin):
        '''unique_token_hin is the third dimension in one hot encoding or no of tokens in hindi or output size'''
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.cell = cell
        self.bi_dir = bi_dir
        self.unique_tokens_hin=unique_tokens_hin
        self.embed_dim=embed_dim
        self.drop_out=drop_out
        '''The input to the embedding will be of shape (batch_size,sequence_length) and the output size will be
        (batch_size,seq_length,embed_size)'''

        self.embedding = nn.Embedding(unique_tokens_hin, embed_dim) 
        self.drop__out=nn.Dropout(p=self.drop_out)
        self.relu=nn.ReLU()
        self.rnn=nn.RNN(embed_dim,hidden_dim,dropout=self.drop_out,
                        num_layers=layer_dim,bidirectional=bi_dir,batch_first=True)
        self.gru=nn.GRU(embed_dim,hidden_dim,dropout=self.drop_out,
                        num_layers=layer_dim,bidirectional=bi_dir,batch_first=True)
        self.lstm=nn.LSTM(embed_dim,hidden_dim,dropout=self.drop_out,
                         num_layers=layer_dim,bidirectional=bi_dir,batch_first=True)
        self.out_put = nn.Linear((1 + int(self.bi_dir)) * self.hidden_dim, self.unique_tokens_hin)
        #number of unique tokens in hindi is the output dimension of decoder layer
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x,hidden):
        '''Here, x is the decoder input data'''
        batch_size=x.size(0)
        embedding_output = self.embedding(x.long())
        embedding_output=embedding_output.to(device)
        #embedding_output = self.drop__out(embedding_output)
        if self.cell=="GRU":
            hidden=hidden.contiguous()
            output,h_n=self.gru(embedding_output,hidden)
            output=self.relu(output)
            output=self.softmax(self.out_put(output))
            return output,h_n
        elif self.cell=='RNN':
            output,h_n=self.gru(embedding_output,hidden)
            output=self.relu(output)
            output=self.softmax(self.out_put(output))
            return output,h_n
        elif self.cell=='LSTM':
            h0=hidden[0].contiguous()
            c0=hidden[1].contiguous()
            output,(h_n,c_n)=self.lstm(embedding_output,(h0,c0))
            output=self.relu(output)
            output=self.softmax(self.out_put(output))
            return output,(h_n,c_n)
            
    def decoder_initial(self,batch_size,device=device):
        if self.cell == "LSTM":
            h_0 =torch.randn((1 + int(self.bi_dir)) * self.layer_dim, batch_size, self.hidden_dim, device=device)
            c_0=torch.randn((1 + int(self.bi_dir)) * self.layer_dim, batch_size, self.hidden_dim, device=device) 
            return (h_0,c_0)
          #H_0,C_0 HAVE SAME DIMENSION
        else:
            h_0=torch.randn((1 + int(self.bi_dir)) * self.layer_dim, batch_size, self.hidden_dim, device=device)
            return h_0
        


# In[57]:


'''To use when the number of encoder and decoder layers are different'''
class Reshape(nn.Module):
    def __init__(self, num_enc_layers, num_dec_layers, cell, bi_dir):
        super(Reshape, self).__init__()
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.cell = cell
        self.bi_dir = bi_dir
        self.linear = nn.Linear(num_enc_layers * int(1 + bi_dir), num_dec_layers * int(1 + bi_dir))

    def forward(self, h_n_enc):
        if self.cell == 'LSTM':
            x=x.permute(*torch.arange(x.ndim - 1, -1, -1))
            h_dec = self.linear(h_n_enc[0].permute(*torch.arange(h_n_enc.ndim - 1, -1, -1)))
            c_dec = self.linear(h_n_enc[1].permute(*torch.arange(c_n_enc.ndim - 1, -1, -1)))
            h_0_dec = (h_dec.permute(*torch.arange(h_dec.ndim - 1, -1, -1)),
                       c_dec.permute(*torch.arange(c_dec.ndim - 1, -1, -1)))
        else:
            h_0_dec = self.linear(h_n_enc.permute(*torch.arange(h_n_enc.ndim - 1, -1, -1)))
            h_0_dec = h_0_dec.permute(*torch.arange(h_0_dec.ndim - 1, -1, -1))
        return h_0_dec


# In[58]:


def gradient(input_tensor, target_tensor,target_onehot, encoder_model, decoder_model, encoder_optimizer,
          decoder_optimizer,hidden_dim,criterion,input_length,target_length,batch_size,
             teacher_forcing_ratio,num_enc_layers,num_dec_layers,bi_dir,cell,reshape,ropt,device=device):
    
    
    h_0_enc = encoder_model.encoder_initial(batch_size)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    #ropt.zero_grad()
    uh=target_onehot.shape[-1]
    decoder_predicted=torch.zeros(batch_size,target_length,uh,device=device)
    loss = 0
    
    encoder_model.train()
    decoder_model.train()
    #reshape.train()
    
    '''Encoder model is given and the representation of input word is taken from it'''
    for i in range(input_length):
        output_enc, h_n_enc = encoder_model(input_tensor[:,i].unsqueeze(1), h_0_enc)
        h_0_enc=h_n_enc

    dec_input = torch.ones(batch_size,1,device=device)#start token is given as the input and the indices is 1.
    #h_0_dec=reshape(h_n_enc)
    h_0_dec=h_n_enc
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for i in range(target_length):
            output_dec, h_n_dec = decoder_model(dec_input, h_0_dec)
            decoder_predicted[:, i, :] = output_dec.squeeze(1)
            loss += criterion(output_dec.reshape(-1,uh).float(), target_onehot[:,i:i+1,:].reshape(-1,uh).float())
            #loss+=criterion(output_dec.view(-1,uh).float(),target_tensor[:,i].long())
            dec_input = target_tensor[:,i].unsqueeze(1)  # Teacher forcing
            h_0_dec = h_n_dec
            

    else:
        # Without teacher forcing: use its own predictions as the next input
        for i in range(target_length):
            output_dec, h_n_dec = decoder_model(dec_input, h_0_dec)
            #top_values, top_indices = output_dec.topk(k=1, dim=2)
            #dec_input = top_indices.view(-1,1).detach()# detach from history as input
            dec_input = torch.argmax(output_dec,dim=-1)
            decoder_predicted[:, i, :] = output_dec.squeeze(1)
            loss += criterion(output_dec.reshape(-1,uh).float(), target_onehot[:,i:i+1,:].reshape(-1,uh).float())
            #loss+=criterion(output_dec.view(-1,uh).float(),target_tensor[:,i].long())
            h_0_dec=h_n_dec
    
            
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    #ropt.step()
    
    loss_batch = loss.item() / target_length
    char_acc_batch = calculate_char_accuracy(decoder_predicted,target_onehot)
    word_acc_batch = calculate_word_accuracy(decoder_predicted,target_onehot)
    return loss_batch, char_acc_batch, word_acc_batch


# In[59]:


def testing(input_tensor, target_tensor,target_onehot, encoder_model, decoder_model,
          hidden_dim,criterion,input_length,target_length,batch_size,
             num_enc_layers,num_dec_layers,bi_dir,cell,reshape,device=device):
    
    
    h_0_enc = encoder_model.encoder_initial(batch_size)
    uh=target_onehot.shape[-1]
    decoder_predicted=torch.zeros(batch_size,target_length,uh,device=device)
    loss = 0
    
    encoder_model.eval()
    decoder_model.eval()
    #reshape.eval()
    
    '''Encoder model is given and the representation of input word is taken from it'''
    for i in range(input_length):
        output_enc, h_n_enc = encoder_model(input_tensor[:,i].unsqueeze(1), h_0_enc)
        h_0_enc=h_n_enc

    dec_input = torch.ones(batch_size,1,device=device)#start token is given as the input and the indices is 1.
    #h_0_dec=reshape(h_n_enc)
    h_0_dec=h_n_enc
    
    # Without teacher forcing: use its own predictions as the next input
    for i in range(target_length):
        output_dec, h_n_dec = decoder_model(dec_input, h_0_dec)
        #top_values, top_indices = output_dec.topk(k=1, dim=2)
        #dec_input = top_indices.view(-1,1).detach()# detach from history as input
        dec_input = torch.argmax(output_dec,dim=-1)
        decoder_predicted[:, i, :] = output_dec.squeeze(1)
        h_0_dec=h_n_dec
        loss += criterion(output_dec.reshape(-1,uh).float(), target_onehot[:,i:i+1,:].reshape(-1,uh).float())
        #loss+=criterion(output_dec.view(-1,uh).float(),target_tensor[:,i].long())
    
    loss_batch = loss.item() / target_length
    char_acc_batch = calculate_char_accuracy(decoder_predicted,target_onehot)
    word_acc_batch = calculate_word_accuracy(decoder_predicted,target_onehot)
    return loss_batch, char_acc_batch, word_acc_batch


# In[60]:


def train(x,y,yonehot,x_val,y_val,yonehot_val,epochs,optimizer,learning_rate,weight_decay,layer_dim,bi_dir,
          teacher_forcing_ratio,cell,embed_dim,hidden_dim,batch_size,drop_out,device=device):
    
    start = time.time()
    input_length = x.shape[1]
    target_length = y.shape[1]
    num_enc_layers=layer_dim
    num_dec_layers=layer_dim
    
    encoder_model = Encoder(embed_dim, hidden_dim, num_enc_layers, drop_out, bi_dir, cell,unique_tokens_eng).to(device)
    decoder_model = Decoder(embed_dim, hidden_dim, num_dec_layers, drop_out, bi_dir, cell,unique_tokens_hin).to(device)
    reshape=Reshape(num_enc_layers, num_dec_layers, cell, bi_dir).to(device)
    
    if optimizer=='Adam':
        encoder_optimizer=torch.optim.Adam(encoder_model.parameters(),lr=learning_rate,weight_decay=weight_decay)
        decoder_optimizer=torch.optim.Adam(decoder_model.parameters(),lr=learning_rate,weight_decay=weight_decay)
        ropt = torch.optim.Adam(reshape.parameters(),lr=learning_rate,weight_decay=weight_decay)
    elif optimizer=='NAdam':
        encoder_optimizer=torch.optim.NAdam(encoder_model.parameters(),lr=learning_rate,weight_decay=weight_decay)
        decoder_optimizer=torch.optim.NAdam(decoder_model.parameters(),lr=learning_rate,weight_decay=weight_decay)
        ropt = torch.optim.NAdam(reshape.parameters(),lr=learning_rate,weight_decay=weight_decay)
    elif optimizer=='SGD':
        
        encoder_optimizer=torch.optim.SGD(encoder_model.parameters(),lr=learning_rate,weight_decay=weight_decay)
        decoder_optimizer=torch.optim.SGD(decoder_model.parameters(),lr=learning_rate,weight_decay=weight_decay)
        ropt = torch.optim.SGD(reshape.parameters(),lr=learning_rate,weight_decay=weight_decay)
        
    train_loader = data_loader(x,y,yonehot,batch_size,device=device)
    val_loader = data_loader(x_val,y_val,yonehot_val,batch_size,device=device)
    
    criterion = nn.CrossEntropyLoss()
    #criterion=nn.NLLLoss()
    
    epoch_losses=[]
    epoch_char_accuracy=[]
    epoch_word_accuracy=[]
    epoch_losses_val=[]
    epoch_char_accuracy_val=[]
    epoch_word_accuracy_val=[]
    
    for i in range(1, epochs + 1):
        batch_losses=[]
        batch_char_acc=[]
        batch_word_acc=[]
        batch_losses_val=[]
        batch_char_acc_val=[]
        batch_word_acc_val=[]
        for enc_input_tensor,dec_target_tensor,dec_onehot in train_loader:
            
            loss_batch,char_acc_batch,word_acc_batch = gradient(enc_input_tensor,dec_target_tensor,dec_onehot,
                    encoder_model,decoder_model,encoder_optimizer,decoder_optimizer,
                    hidden_dim,criterion,input_length,target_length,batch_size, teacher_forcing_ratio,
                            num_enc_layers,num_dec_layers,bi_dir,cell,reshape,ropt,device=device)
            
        for enc_input_tensor_val,dec_target_tensor_val,dec_onehot_val in val_loader: 
            
            loss_batch_val,char_acc_batch_val,word_acc_batch_val=testing(enc_input_tensor_val,dec_target_tensor_val,
                        dec_onehot_val,encoder_model,decoder_model,hidden_dim,criterion,input_length,
                        target_length,batch_size,num_enc_layers,num_dec_layers,bi_dir,cell,reshape,device=device)
          
            batch_losses.append(loss_batch)
            batch_char_acc.append(char_acc_batch)
            batch_word_acc.append(word_acc_batch)
            batch_losses_val.append(loss_batch_val)
            batch_char_acc_val.append(char_acc_batch_val)
            batch_word_acc_val.append(word_acc_batch_val)
            
            
        epoch_loss=sum(batch_losses)/len(batch_losses)
        epoch_char_acc=sum(batch_char_acc)/len(batch_char_acc)
        epoch_word_acc=sum(batch_word_acc)/len(batch_word_acc)
        epoch_losses.append(epoch_loss)
        epoch_char_accuracy.append(epoch_char_acc)
        epoch_word_accuracy.append(epoch_word_acc)
        
        epoch_loss_val = sum(batch_losses_val) / len(batch_losses_val)
        epoch_char_acc_val = sum(batch_char_acc_val) / len(batch_char_acc_val)
        epoch_word_acc_val = sum(batch_word_acc_val) / len(batch_word_acc_val)
        epoch_losses_val.append(epoch_loss_val)
        epoch_char_accuracy_val.append(epoch_char_acc_val)
        epoch_word_accuracy_val.append(epoch_word_acc_val)
        
        wandb.log({'train_loss': epoch_loss, 'train_char_acc': epoch_char_acc, 'train_word_acc': epoch_word_acc, 'valid_loss': epoch_loss_val, 'valid_char_acc': epoch_char_acc_val, 'valid_word_acc': epoch_word_acc_val})
        print(f'{timeSince(start, i / epochs)} ({i} {i / epochs * 100:.2f}%) Trainloss: {epoch_losses[-1]:.4f} Char Accuracy: {epoch_char_accuracy[-1]:.4f} Word Accuracy: {epoch_word_accuracy[-1]:.4f}')
        print(f'{timeSince(start, i / epochs)} ({i} {i / epochs * 100:.2f}%) Validationloss: {epoch_losses_val[-1]:.4f} Char Accuracy: {epoch_char_accuracy_val[-1]:.4f} Word Accuracy: {epoch_word_accuracy_val[-1]:.4f}')


    return encoder_model,decoder_model,encoder_optimizer,decoder_optimizer
    


# In[61]:


# Sample wandb run()
def wandb_run():
    
    config_defaults = {
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'epochs': 10,
        'hidden_dim': 512,
        'embed_dim': 512,
        'layer_dim': 2,
        'drop_out': 0.3,
        'cell': 'GRU',
        'weight_decay': 0.001,
        'bi_dir': True,
        'batch_size': 64
    }
    wandb.init(config=config_defaults)
    config = wandb.config
    run_name = f'lr_{config.learning_rate}_acti_{config.optimizer}_epochs_{config.epochs}_cell_{config.cell}_dir_{config.bi_dir}_num_hid_{config.hidden_dim}_ld_{config.layer_dim}_ed_{config.embed_dim}__drop__{config.drop_out}'
    print(run_name)
    wandb.init(name=run_name)
    encoder_model, decoder_model, encoder_optimizer, decoder_optimizer = train(
        x=enc_input_data,
        y=dec_output_data,
        yonehot=decoder_output_data,
        x_val=enc_input_data_val,
        y_val=dec_output_data_val,
        yonehot_val=decoder_output_data_val,
        epochs=config.epochs,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        layer_dim=config.layer_dim,
        bi_dir=config.bi_dir,
        teacher_forcing_ratio=0.5,
        cell=config.cell,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        batch_size=config.batch_size,
        drop_out=config.drop_out
    )
     
    wandb.run.name = run_name
    wandb.run.finish()


# In[62]:


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project',default="CS6910-ASSIGNMENT_3")
    parser.add_argument('--learning_rate', type=float, default =0.001,help='lr value for optimizer')
    parser.add_argument('--hidden_dim', type=float, default=256)
    parser.add_argument('--layer_dim', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--weight_decay',type=float, default=0.0001, help='decay in optim')
    parser.add_argument('--embed_dim', default=256,help='embedding dimension')
    parser.add_argument('--epochs', default='10')
    parser.add_argument('--optimizer', default='Adam',help='optimizers (Adam or Nadam or SGD)')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--cell', type=str, default='LSTM')
    parser.add_argument('--drop_out', type=float, default=0.2)
    parser.add_argument('--bi_dir', type=bool, default=True)

    

    args = parser.parse_args()
    path_train ="/home/agcl/Downloads/hin_train.csv"
    path_test="/home/agcl/Downloads/hin_test.csv"
    path_validation="/home/agcl/Downloads/hin_valid.csv"
    language_names = ['English','transliteration_in_hindi']
    df_train=load_data(path_train,language_names)
    df_validation=load_data(path_validation,language_names)
    df_test=load_data(path_test,language_names)
    english_vocab=split_words(df_train[language_names[0]])
    hindi_vocab=split_words(df_train[language_names[1]])
    int2char_eng=int_to_char(english_vocab)
    char2int_eng=char_to_int(int2char_eng)
    int2char_hin=int_to_char(hindi_vocab)
    char2int_hin=char_to_int(int2char_hin)
    length_eng=[len(i) for i in df_train[language_names[0]]]
    length_hin=[len(i) for i in df_train[language_names[1]]]
    length_eng_max=(max(length_eng)) + 2
    length_hin_max=(max(length_hin)) + 2
    num_english_tokens=len(english_vocab)
    num_hindi_tokens=len(hindi_vocab)
    unique_tokens_eng=len(english_vocab)
    unique_tokens_hin=len(hindi_vocab)
    enc_input_data,dec_input_data,dec_output_data = process_data(df_train,english_vocab=english_vocab,
                hindi_vocab=hindi_vocab,length_eng_max=length_eng_max,length_hin_max=length_hin_max,
                char2int_eng=char2int_eng,char2int_hin=char2int_hin)
    enc_input_data_val,dec_input_data_val,dec_output_data_val = process_data(df_validation,english_vocab=english_vocab,
                hindi_vocab=hindi_vocab,length_eng_max=length_eng_max,length_hin_max=length_hin_max,
                char2int_eng=char2int_eng,char2int_hin=char2int_hin)
    enc_input_data_test,dec_input_data_test,dec_output_data_test = process_data(df_test,english_vocab=english_vocab,
                hindi_vocab=hindi_vocab,length_eng_max=length_eng_max,length_hin_max=length_hin_max,
                char2int_eng=char2int_eng,char2int_hin=char2int_hin)
                
    encoder_input_data,decoder_input_data,decoder_output_data=one_hot_encoding(df_train,english_vocab=english_vocab,
                hindi_vocab=hindi_vocab,length_eng_max=length_eng_max,length_hin_max=length_hin_max,
                char2int_eng=char2int_eng,char2int_hin=char2int_hin)
    encoder_input_data_val,decoder_input_data_val,decoder_output_data_val=one_hot_encoding(df_validation,english_vocab=english_vocab,
                hindi_vocab=hindi_vocab,length_eng_max=length_eng_max,length_hin_max=length_hin_max,
                char2int_eng=char2int_eng,char2int_hin=char2int_hin)
    encoder_input_data_test,decoder_input_data_test,decoder_output_data_test=one_hot_encoding(df_test,english_vocab=english_vocab,
                hindi_vocab=hindi_vocab,length_eng_max=length_eng_max,length_hin_max=length_hin_max,
                char2int_eng=char2int_eng,char2int_hin=char2int_hin)
    enc_input_data=enc_input_data.long()
    dec_input_data=dec_input_data.long()
    enc_input_data_test=enc_input_data_test.long()
    dec_input_data_test=dec_input_data_test.long()
    enc_input_data_val=enc_input_data_val.long()
    dec_input_data_val=dec_input_data_val.long()
    encoder_input_data=encoder_input_data.long()
    decoder_input_data=decoder_input_data.long()
    decoder_output_data=decoder_output_data.long()
    unique_tokens_eng=len(english_vocab)
    unique_tokens_hin=len(hindi_vocab)
    wandb_run()


# In[ ]:




