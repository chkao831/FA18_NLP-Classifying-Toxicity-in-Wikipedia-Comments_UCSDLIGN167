#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import bcolz as bz
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
from torch.autograd import Variable


# In[2]:


def divide_string(sentence):
    s_l = sentence.split()
    r_l = []
    punc = [ '?', '!', '*']
    delete = ['%','^','&','_','>','<','\\','$', '+', '~','`', ',', '.', ';',
              '#', '@', '(', ')', '[' , ']', '/', '|', '{', '}', '"', '-', '=']
    for i in range(len(s_l)):
        tmp = 0
        flag = True;
        for j in range(len(s_l[i])):
            if s_l[i][j] in punc:
                if(flag):
                    r_l.append(s_l[i][tmp : j])
                tmp = j
                flag = True
            if s_l[i][j] in delete:
                if(flag):
                    r_l.append(s_l[i][tmp : j])
                tmp = j
                flag = False
            else:
                if(not flag):
                    tmp = tmp + 1;
                flag = True
        if(flag):
            r_l.append(s_l[i][ tmp : len(s_l[i]) ] )
    str_list = [x for x in r_l if x != '']
    return str_list


# In[3]:


a = "THIS is a go(%*#od day!===thanks $%@%*(don't}}} know... ====================================="
b = divide_string(a)
print(len(b))
print(b)


# In[4]:



vectors = bz.open(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\glove\27B.100.dat")[:]
words = pickle.load(open(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\glove\27B.100_words.pkl", 'rb'))
word2idx = pickle.load(open(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\glove\27B.100_idx.pkl", 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}


# In[5]:


# sents is a list of string
def remove_infrequent_words(sents):
    word_counts = {}
    divide_sentence = []
    #divide each sentence first
    for s in sents:
        tmp = divide_string(s)
        divide_sentence.append(tmp)
    
    # count the word
    for s in divide_sentence:
        for w in s:
            if w in word_counts:
                word_counts[w] += 1
            else:
                word_counts[w] = 1

    threshold = 20
    filtered_sents = []
    for s in divide_sentence:
        new_s = []
        for w in s:
            if word_counts[w] < threshold:
                new_s.append('<UNKOWN>')
            else:
                new_s.append(w)
        filtered_sents.append(new_s)
    return filtered_sents


# In[6]:


s = ["This is a good time!!", "You are pretty@$&^$@ happy", "test", "test*%& test"]
def word_embedding_glove_total(sentence):
    #create the total size of weight matrix
    weights_matrix = np.zeros((len(sentence), 100, 103))
    
    
    processed_sentence = remove_infrequent_words(sentence)
    
    for k in range(len(processed_sentence)):
        target_vocab = processed_sentence[k]
        emb_dim = 100

        # enumerate the word, break at position 100
        for i, word in enumerate(target_vocab):
            if i == 100:
                break;
            try: 
                tmp_vec = glove[word]
            except KeyError:
                tmp_vec = np.random.normal(scale=0.6, size=(emb_dim, ))
                tmp_dic = {word:tmp_vec}
                glove.update(tmp_dic)
            #write something to change it to 103
            ind = 0
            if word.isupper():
                tmp_vec = np.append(tmp_vec, [1])
            else:
                tmp_vec = np.append(tmp_vec, [0])
                ind = ind + 1
    
            if word.islower():
                tmp_vec = np.append(tmp_vec, [1])
            else:
                tmp_vec = np.append(tmp_vec, [0])
                ind = ind + 1
    
            if ind == 2:
                tmp_vec = np.append(tmp_vec, [1])
            else:
                tmp_vec = np.append(tmp_vec, [0])
    
            weights_matrix[k][i] = tmp_vec;
    return weights_matrix

test =  word_embedding_glove_total(s)
print(test.shape)
sentences = remove_infrequent_words(s)
print(sentences)


# In[7]:


s = ["This is a good time!!", "You are pretty@$&^$@ happy", "test", "test*%& test"]
def base_glove_total(sentence):
    #create the total size of weight matrix
    weights_matrix = np.zeros((len(sentence),100, 103))
    
    
    for k in range(len(sentence)):
        target_vocab = sentence[k].split()
        emb_dim = 100

        # enumerate the word, break at position 100
        for i in range(len(target_vocab)):
            if i == 100:
                break;
            try: 
                word = target_vocab[i]
                tmp_vec = glove[target_vocab[i]]
            except KeyError:
                tmp_vec = np.random.normal(scale=0.6, size=(emb_dim, ))
                
            #write something to change it to 103
            ind = 0
            if word.isupper():
                tmp_vec = np.append(tmp_vec, [1])
            else:
                tmp_vec = np.append(tmp_vec, [0])
                ind = ind + 1
    
            if word.islower():
                tmp_vec = np.append(tmp_vec, [1])
            else:
                tmp_vec = np.append(tmp_vec, [0])
                ind = ind + 1
    
            if ind == 2:
                tmp_vec = np.append(tmp_vec, [1])
            else:
                tmp_vec = np.append(tmp_vec, [0])
    
            weights_matrix[k][i] = tmp_vec;
    return weights_matrix

test =  base_glove_total(s)
print(test.shape)


# In[8]:


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class tcCNN(nn.Module):
    def __init__(self, batch_size):
        super(tcCNN, self).__init__()

        self.batch_size = batch_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5,103), stride=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.bn_layer = nn.BatchNorm2d(64)
        self.fc = nn.Sequential(nn.Linear(64*1*96, 50),
                                nn.Dropout())
        self.out_layer = nn.Linear(50,6)
        

        
    def forward(self, weights_matrix):
        #didn't use batch, may need to add batch size in the middle
        out = self.layer1(weights_matrix)
        out = self.bn_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        y_pred = self.out_layer(out)
        
        return torch.sigmoid(y_pred)
                                    
    


# In[9]:


data = pd.read_csv(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\data\train.csv", encoding = "ISO-8859-1")
#data.head()


# In[10]:


x_train = data['comment_text'] 
y_train = data.iloc[:,2:8]
print(x_train.shape)
print(y_train.shape)


# In[11]:


data = pd.read_csv(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\data\test.csv", encoding = "ISO-8859-1")
x_test = data['comment_text']
print(x_test.shape)
data = pd.read_csv(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\data\test_labels.csv", encoding = "ISO-8859-1")
y_test = data.iloc[:,1:7]
print(y_test.shape)


# In[12]:


#the embedding
start = time.time()
train_set = word_embedding_glove_total(x_train)
#train_set = base_glove_total(x_train)
print(len(glove))
end = time.time()
print(end - start)


# In[13]:


print(train_set.shape)
w = train_set[512:1024,:,:]
train_set_s = train_set[:30000,:,:]
validation_x = train_set[30000:40000,:,:].reshape((10000,1,100,103))
print(train_set_s.shape)
print(validation_x.shape)


# In[14]:


data = pd.read_csv(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\data\test.csv", encoding = "ISO-8859-1")
data2 = pd.read_csv(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\data\test_labels.csv", encoding = "ISO-8859-1")
df = pd.concat([data.reset_index(drop=True), data2.reset_index(drop=True)], axis=1)
df = df.drop(['id', 'id'],axis=1)
df = df[df.toxic != -1]
print(data.shape)
print(data2.shape)
print(df.shape)
x_test = df.reset_index()['comment_text']
y_test = df.reset_index().iloc[:, 2:8]
print(x_test.shape)
print(y_test.shape)


# In[15]:


batch_size = 512
epoch = 5
lr = 0.005
output_dim = 6
input_dim = 103
#train_size = len(x_train)
train_size = 30000
train_output = y_train.values
validation_y = train_output[30000:40000]
print(validation_y.shape)


model = tcCNN(batch_size = batch_size)
model = model.cuda()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# In[16]:


#then do the testing
#first test on 10000 set


def evaluation(test_set, test_output):
    correct = 0
    total = len(test_output)
    with torch.no_grad():
        x_t = torch.cuda.FloatTensor(test_set)
        y_t = torch.cuda.FloatTensor(test_output)
        y_pred = model(x_t)
        y_pred = y_pred.reshape((total, 6))
        y_pred_c = y_pred > 0.6
        y_pred_c = y_pred_c.type(torch.cuda.FloatTensor)
        for i in range(total):
            if torch.equal(y_pred_c[i], y_t[i]):
                correct = correct + 1
        print(correct)


# In[17]:




start = time.time()

#Training

for i in range(epoch):  # again, normally you would NOT do 300 epochs, it is toy data
    j = 0
    total_loss = 0
    while(j < train_size ):
        k = j + 512;
        if k > train_size:
            break
        
        #prepare input and output
        x_t = train_set_s[j:k,:,:]
        x_t = x_t.reshape(512,1,100,103)
        x_t = torch.cuda.FloatTensor(x_t)
        y_t = train_output[j:k,:]
        #y_t = y_t.sum(axis=1)
        y_t = torch.cuda.FloatTensor(y_t)
        
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()



        # Step 2. Run our forward pass.
        y_pred = model(x_t)
        y_pred = y_pred.reshape((512,6))
        
        
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_fn(y_pred, y_t)
        total_loss += loss
        loss.backward()
        optimizer.step()
        j = k
        evaluation(validation_x, validation_y)
    print(total_loss)
        
        
end = time.time()
print(end - start)


# In[18]:


#test set
total = 10000
test_set_d = x_test[0:total]
test_output = y_test.values[0:total, :]
test_set = word_embedding_glove_total(test_set_d)
#test_set = base_glove_total(test_set_d)
test_set = test_set.reshape((total,1,100,103))
print(len(glove))


# In[19]:


evaluation(test_set, test_output)


# In[ ]:




