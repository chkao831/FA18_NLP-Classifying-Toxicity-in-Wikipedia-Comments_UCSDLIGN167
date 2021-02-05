#!/usr/bin/env python
# coding: utf-8

# In[9]:


import bcolz as bz
import pickle
import numpy as np

words = []
idx = 0
word2idx = {}
vectors = bz.carray(np.zeros(1), rootdir=r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\glove\27B.100.dat", mode='w')

with open(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\glove\glove.twitter.27B.100d.txt", 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        if(len(vect) != 100):
            b = np.zeros(1)
            vect = np.concatenate((vect,b))
        vectors.append(vect)
    
vectors = bz.carray(vectors[1:].reshape((idx, 100)), rootdir=r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\glove\27B.100.dat", mode='w')
vectors.flush()
pickle.dump(words, open(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\glove\27B.100_words.pkl", 'wb'))
pickle.dump(word2idx, open(r"C:\Users\mul02\Desktop\Course\LIGN 167\Final Project\glove\27B.100_idx.pkl", 'wb'))


# In[8]:


import numpy as np
a = np.zeros(10)
b = np.zeros(1)
print(b.shape)
print(a.shape)
c = np.concatenate((a,b))
print(c.shape)


# In[ ]:




