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


import numpy as np
result = {"a": 1.1181753789488595, 'b': 0.5566080288678394, 'c': 0.4718269778030734, 'd': 0.48716683119447185, 'e': 1.0, 'f': 0.1395076201641266, 'g': 0.20941558441558442}

names = ['id','data']
formats = ['str','f8']
dtype = dict(names = names, formats=formats)
array = np.array(list(result.items()), dtype=dtype)

print(array)


# In[3]:


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


# In[4]:


a = "THIS is a go(%*#od day!===thanks $%@%*(don't}}} know... ====================================="
b = divide_string(a)
print(len(b))
print(b)


# In[5]:



vectors = bz.open(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\glove\27B.100.dat")[:]
words = pickle.load(open(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\glove\27B.100_words.pkl", 'rb'))
word2idx = pickle.load(open(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\glove\27B.100_idx.pkl", 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}


# In[6]:


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
                new_s.append('<UNKNOWN>')
            else:
                new_s.append(w)
        filtered_sents.append(new_s)
    return filtered_sents


# In[7]:


s = ["This is a good time!!", "You are pretty@$&^$@ happy", "test", "test*%& test"]
def prep_embedding_weight(sentence):
    #create the total size of weight matrix
    emb_dim = 100
    processed_sentence = remove_infrequent_words(sentence)
    
    vocab_size = 1;
    word_index = {"PADDING_STRING":0}
    
    #count the unique word that appares
    for k in range(len(processed_sentence)):
        target_vocab = processed_sentence[k]
        for i,word in enumerate(target_vocab):
            if word in word_index:
                a = 1;
            else:
                word_index[word] = vocab_size
                vocab_size += 1
                
    #error detector
    if(vocab_size != len(word_index)):
        print("Oops, Something wrong")
        return
    
    
    #create the pre-train weight matrix
    weights_matrix = np.zeros((vocab_size,100))
    unrecogized_word = 0;
    for key in word_index:
        try: 
            tmp_vec = np.random.normal(scale=0.6, size=(emb_dim, ))
        except KeyError:
            tmp_vec = np.random.normal(scale=0.6, size=(emb_dim, ))
        i = word_index[key]
        weights_matrix[i] = tmp_vec;
    #set the padding vector to be all zero
    weights_matrix[0] = np.zeros((emb_dim,))
    return weights_matrix, word_index

test, word =  prep_embedding_weight(s)
print(test.shape)
print(word["PADDING_STRING"])


# In[8]:


def prep_embedding_input(sentence, word_index):
    emb_input = np.zeros((len(sentence),100))
    processed_sentence = remove_infrequent_words(sentence)
    for k in range(len(processed_sentence)):
        target_vocab = processed_sentence[k]
        for i,word in enumerate(target_vocab):
            if i == 100:
                break
            try:
                value = word_index[word]
            except:
                value = word_index["<UNKNOWN>"]
            emb_input[k][i] = value
    return emb_input


# In[9]:


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class tcCNN(nn.Module):
    def __init__(self, batch_size, vocab_size, pretrained_weight):
        super(tcCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        
        
        self.batch_size = batch_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5,100), stride=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.bn_layer = nn.BatchNorm2d(64)
        self.fc = nn.Sequential(nn.Linear(64*1*96, 50),
                                nn.Dropout())
        self.out_layer = nn.Linear(50,6)
        

        
    def forward(self, sentences):
        #didn't use batch, may need to add batch size in the middle
        out = self.embedding(sentences)
        out = self.layer1(out)
        out = self.bn_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        y_pred = self.out_layer(out)
        
        return torch.sigmoid(y_pred)
                                    
    


# In[10]:


data = pd.read_csv(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\data\train.csv", encoding = "ISO-8859-1")
#data.head()


# In[11]:


x_train = data['comment_text'] 
y_train = data.iloc[:,2:8]
print(x_train.shape)
print(y_train.shape)


# In[12]:


data = pd.read_csv(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\data\test.csv", encoding = "ISO-8859-1")
x_test = data['comment_text']
print(x_test.shape)
data = pd.read_csv(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\data\test_labels.csv", encoding = "ISO-8859-1")
y_test = data.iloc[:,1:7]
print(y_test.shape)


# In[13]:


#the embedding
start = time.time()
weights_matrix, word_index = prep_embedding_weight(x_train)
train_set = prep_embedding_input(x_train, word_index)

print(train_set.shape)
print(weights_matrix.shape)
print(len(word_index))
end = time.time()
print(end - start)


# In[14]:


train_set_s = train_set[:30000,:]
validation_x = train_set[30000:40000,:].reshape((10000,1,100))
print(train_set_s.shape)
print(validation_x.shape)


# In[15]:


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


# In[16]:


batch_size = 512
epoch = 200
lr = 0.005
output_dim = 6
input_dim = 100
#train_size = len(x_train)
train_size = 30000
train_output = y_train.values
validation_y = train_output[30000:40000]
vocab_size = (len(word_index))
print(validation_y.shape)


model = tcCNN(batch_size = batch_size,vocab_size=vocab_size, pretrained_weight=weights_matrix)
model = model.cuda()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# In[17]:


#then do the testing
#first test on 10000 set


def evaluation(test_set, test_output):
    correct = 0
    total = len(test_output)
    with torch.no_grad():
        x_t = torch.cuda.LongTensor(test_set)
        y_t = torch.cuda.FloatTensor(test_output)
        y_pred = model(x_t)
        y_pred = y_pred.reshape((total, 6))
        y_pred_c = y_pred > 0.6
        y_pred_c = y_pred_c.type(torch.cuda.FloatTensor)
        for i in range(total):
            if torch.equal(y_pred_c[i], y_t[i]):
                correct = correct + 1
        print(correct)


# In[18]:




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
        x_t = train_set_s[j:k,:]
        x_t = x_t.reshape(512,1,100)
        x_t = torch.cuda.LongTensor(x_t)
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


# In[19]:


#test set
total = 10000
test_set_d = x_test[10000:20000]
test_output = y_test.values[10000:20000, :]
test_set = prep_embedding_input(test_set_d, word_index)
#test_set = base_glove_total(test_set_d)
test_set = test_set.reshape((total,1,100))
print(len(glove))


# In[20]:


evaluation(test_set, test_output)


# In[ ]:




