#!/usr/bin/env python

###import texts 

import nltk
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from transformers import BertTokenizer, BertForNextSentencePrediction

#from nltk.tokenize import RegexpTokenizer
filelist=["Janus", "shaka", "doping", "thy", "worlden", "monocole", "winetaste", "orangejuice", "beekeeping", "nationalflag", "union", "vr"]

appended_data = []

for j in range(0, len(filelist)):
    
    filename = filelist[j]
    file=open(filename, "r", encoding="utf-8")
    text=file.read()
    sentences = nltk.sent_tokenize(text)

    # Join the sentences with a newline character
    formatted_text = "\n".join(sentences)

    ##divide text into sentences

    sentences = [p for p in formatted_text.split('\n') if p]

    ##transformer

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-multilingual-cased')


    ## contextual sentences are all sentences in the left to the target sentence
    val = [] # Create an empty list
    sent_No = [] # create empty list
    
    for i in range(1, len(sentences)):

        text = sentences[:i]
        next_text = sentences[i]

        # Tokenize the input text and next sentence
        tokenized_text = tokenizer.encode(text, next_text, return_tensors='pt')

        # Compute the probability of the next sentence
        outputs = model(tokenized_text)
        probs = outputs[0][0][0].detach().numpy()
        # probs is an array
        pb=probs.item()
        surp=-np.log(pb)
        val.append(surp)

        #val
        No = str(j+1) + '_' + str(i+1)
        sent_No.append(No)  
        df = pd.DataFrame(val, columns=['sent_surp'])
        df['sent_No'] = sent_No#str(j)+list(range(2, (len(sentences)+1)))
        
    appended_data.append(df)
appended_data = pd.concat(appended_data)
#appended_data.to_csv('appended_surp.csv')

##continue to compute sentence similarity

from transformers import AutoTokenizer, AutoModel

app_dat = []
for j in range(0, len(filelist)):
    
    filename = filelist[j]
    file=open(filename, "r", encoding="utf-8")
    text=file.read()
    sentences = nltk.sent_tokenize(text)

    # Join the sentences with a newline character
    formatted_text = "\n".join(sentences)

    ##divide text into sentences

    sentences = [p for p in formatted_text.split('\n') if p]
    
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModel.from_pretrained("bert-base-multilingual-cased")
    num = [] # Create an empty list
    sent_No = []
    for i in range(1, len(sentences)):
        if i > 1:
           sent1 = sentences[(i-3)]
           sent2 = sentences[(i-2)]
           sent3 = sentences[i-1]
   
           inputs1 = tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
           inputs2 = tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)
           inputs3 = tokenizer(sent3, return_tensors='pt', padding=True, truncation=True)
           with torch.no_grad():
              outputs1 = model(**inputs1)
              outputs2 = model(**inputs2)
              outputs3 = model(**inputs3)
           embeddings1 = outputs1.pooler_output
           embeddings2 = outputs2.pooler_output
           embeddings3 = outputs3.pooler_output

           # Compute cosine similarity between the embeddings
           sim1 = cosine_similarity(embeddings1, embeddings2)
           sim2 = cosine_similarity(embeddings1, embeddings3)
           sim3 = cosine_similarity(embeddings2, embeddings3)
           s1=sim1.item()*1/3
           s2=sim2.item()*1/2
           s3=sim3.item()
           sim=s1+s2+s3
           num.append(sim)
           No = str(j+1) + '_' + str(i+1)
           sent_No.append(No)  
           dx = pd.DataFrame(num, columns=['sent_semrev'])
           dx['sent_No'] = sent_No#str(j)+list(range(2, (len(sentences)+1)))
        
    app_dat.append(dx)
app_dat1 = pd.concat(app_dat)
#app_dat1.to_csv('appended_semrev.csv')

###merge two dataframes

dfy = pd.merge(appended_data,app_dat1,on = 'sent_No', how='left')
dfy.to_csv('app_surp_sem.csv')
