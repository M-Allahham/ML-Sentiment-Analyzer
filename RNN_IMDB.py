#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from collections import Counter


# <font color=red>Load data</font>

# In[2]:


df_train = pd.read_csv('data/IMDB_train.csv')


# In[3]:


df_train.head()


# <font color=red> Convert words in sentence to lower case and
# remove non-word characters</font>

# In[4]:


def preprocess_sentence(s):
    # convert to lower
    s = s.lower()
    # remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    return s


# <font color=red> Determine the vocabulary size </font>

# In[5]:


def get_vocabulary_size(reviews):
    reviews = [preprocess_sentence(s) for s in reviews]
    combined_reviews = ' '.join(reviews)
    words = combined_reviews.split()
    count_words = Counter(words)
    total_words = len(words)
    sorted_words = count_words.most_common(total_words)
    return len(sorted_words), sorted_words


# In[6]:


vocab_dim, sorted_words = get_vocabulary_size(df_train['review'].tolist())


# In[7]:


vocab_dim


# <font color=red> Word to integer </font>

# In[8]:


word2int = {w: i+1 for i, (w, c) in enumerate(sorted_words)}


# In[9]:


list(word2int.items())[:5]


# <font color=red> Tokenize the sentence (review)</font>

# In[10]:


def tokenize(reviews):
    int_reviews = []
    for review in reviews:
        review = preprocess_sentence(review)
        r = [word2int[w] for w in review.split()]
        int_reviews.append(r)
    return int_reviews


# In[11]:


int_reviews = tokenize(df_train['review'].tolist())


# In[12]:


# int_reviews[:1]


# <font color=red> Padding sentence by adding 0 at the end to reach the maximum lenght
# or truncating the sentence to meet the maxium length</font>

# In[13]:


def padding_review(int_reviews, maximum_length=200):
    padded_reviews = np.zeros((len(int_reviews), maximum_length), dtype=int)
    for i, int_review in enumerate(int_reviews):
        review_len = len(int_review)
        if review_len >= maximum_length:
            padded_reviews[i, :] = np.array(int_review[:maximum_length])
        else:
            _temp = int_review + list(np.zeros(maximum_length-review_len))
            padded_reviews[i, :] = np.array(_temp)
    return padded_reviews


# In[14]:


maximum_length=200
padded_reviews = padding_review(int_reviews, maximum_length)
vocab_dim = vocab_dim + 1 # +1 for padding 0


# <font color=red> Encode labels: 1-> positive, 0->negative</font>

# In[15]:


int_labels = [1 if label == 'positive' else 0 
              for label in df_train['sentiment'].tolist()
             ]
int_labels = np.array(int_labels)


# <font color=red> Loading data and batching </font>

# In[16]:


train_data = TensorDataset(torch.from_numpy(padded_reviews),
                          torch.from_numpy(np.array(int_labels)))
train_loader = DataLoader(train_data, shuffle=True, batch_size=128)


# <font color=red> Use GPU or CPU</font>

# In[17]:


# CUDA for pytorch
use_cuda = torch.cuda.is_available() # return True if pc has a GPU
#use_cuda = False
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)


# <font color=red> Define a RNN model </font>

# In[18]:


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_dim, output_dim, embedding_dim,
                 hidden_dim, n_layers, dropout_rate):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embbeding layer
        # Ref: https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           dropout=dropout_rate, batch_first=True)
         # dropout layer
        self.dropout = nn.Dropout(0.5)
        # fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size = x.size(0)
        
        #embbeding:
        embed = self.embedding(x)
        # initialized hidden state and cell state
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)) 
        # lSTM
        out, hidden = self.lstm(embed, hidden)
        
        # reshape output size
        out = out.contiguous().view(-1, self.hidden_dim)
        # dropout
        out = self.dropout(out)
        # fully connected layer
        out = self.fc(out)
        # sigmoid
        out = self.sigmoid(out)
        # reshape to get batch_first=True
        out = out.view(batch_size, -1)
        # for Sentiment model, the final output is the output
        #  of the last timestep
        out = out[:, -1]
        
        return out, hidden


# <font color=red> Train model </font>

# In[19]:


output_dim = 1 # positve/negative labels
embedding_dim = 400
hidden_dim = 512
n_layers = 2
model = SentimentLSTM(vocab_dim, output_dim, embedding_dim,
                     hidden_dim, n_layers, 0.5)
model = model.to(device)
loss_function = nn.BCELoss() # BCELoss for binay class instead of using CrossEntropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_epochs = 3

counter = 0
print_every = 100
for epoch in range(n_epochs):
    for inputs, labels in train_loader:
        counter += 1
        labels = labels.to(device)        
        inputs = inputs.type(torch.LongTensor)
        inputs =  inputs.to(device)
        optimizer.zero_grad()
        output, hidden = model(inputs)
        loss = loss_function(output, labels.float())
        
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly
        if counter % print_every == 0:
            print(f'Epoch = {epoch+1}/{n_epochs} ....',
                 f'Step = {counter} ....',
                 f'Binary Loss = {loss.item():.4f}')


# In[24]:


def predict(model, reviews, labels):
    #Tokenize (enocode the reviews)
    correct = 0
    total = 0
    index = 0
    
    _reviews = [preprocess_sentence(s) for s in reviews]
    int_reviews = tokenize(reviews)    
    padded_reviews = padding_review(int_reviews, maximum_length)    
    padded_reviews = torch.from_numpy(padded_reviews)
    padded_reviews = padded_reviews.type(torch.LongTensor)
    padded_reviews = padded_reviews.to(device)
    with torch.no_grad():
        output, hidden = model(padded_reviews)
    for i, o in enumerate(output):
        if o.item() >=0.5:
            if labels[index] == 1:
                correct += 1
                print("Correct")
            total += 1
            index += 1
            #print(f'{reviews[i]}')
            print(f'Positive with probability = {o.item():.3f}')
        else:
            if labels[index] == 0:
                correct += 1
                print("Correct")
            total += 1
            index += 1
            #print(f'{reviews[i]}') 
            print(f'Negative with probability = {o.item():.3f}')
    accuracy = correct / total
    print("Accuracy:", accuracy)


# In[27]:


# fix random seed for reproducible
torch.manual_seed(2021)

df_test = pd.read_csv('data/IMDB_test.csv')

# converting to list
review_list = df_test['review'].tolist()
  
review_list = review_list[:600]

vocab_dim, sorted_words = get_vocabulary_size(df_test['review'].tolist())

word2int = {w: i+1 for i, (w, c) in enumerate(sorted_words)}

int_labels = [1 if label == 'positive' else 0 
              for label in df_test['sentiment'].tolist()
             ]

int_labels = int_labels[:600]

predict(model, review_list, int_labels)


# I only test the network with 600 of the test data entries, as my computer cannot handle any more than that. The network has ~90% accuracy on the training dataset yet only ~58% accuracy on the test dataset, which leads me to believe that overfitting may be occurring. I reduced the number of epochs but it still seems like that is occurring. 

# In[ ]:





# In[ ]:





# In[ ]:




