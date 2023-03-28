#!/usr/bin/env python
#################################
#P(sentence| left context)
###################################
from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F
import math
import nltk
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np



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

    # Initialize the tokenizer and the model
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')


    ## contextual sentences are all sentences in the left to the target sentence
    val = [] # Create an empty list
    sent_No = [] # create empty list
    
    for i in range(1, len(sentences)):
        # Define the left context and the target sentence
        left_context = ' '.join(sentences[:i])
        target_sentence = ' '. join(sentences[i])
        

        # Tokenize the left context and the target sentence separately
        left_context_tokens = tokenizer.tokenize(left_context)
        target_sentence_tokens = tokenizer.tokenize(target_sentence)

        # Initialize a list to store the predicted log probabilities of the individual words
        log_probs =  []

        # Iterate over the target sentence and predict the probability of each word
        prev_token_id = tokenizer.sep_token_id
        
        for token in target_sentence_tokens:
            # Tokenize the current word and concatenate it with the left context and the previously predicted words
            input_tokens = left_context_tokens + [tokenizer.sep_token] + target_sentence_tokens[:target_sentence_tokens.index(token)+1]

            # Convert the input tokens to input IDs
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

            # Generate the attention mask and the segment IDs
            attention_mask = [1] * len(input_ids)
            segment_ids = [0] * len(left_context_tokens) + [1] * (len(input_ids) - len(left_context_tokens))
      
            # Convert the input to PyTorch tensors
            input_ids = torch.tensor([input_ids])
            attention_mask = torch.tensor([attention_mask])
            segment_ids = torch.tensor([segment_ids])

            # Feed the input to the model and get the output
            with torch.no_grad():
               outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
               logits = outputs.logits[0][-1]

            # Compute the probability using a softmax function
            probabilities = F.softmax(logits, dim=-1)

            # Get the probability of the current token
            token_id = tokenizer.convert_tokens_to_ids(token)
            log_prob = probabilities[token_id].log().item()

            # Add the log probability to the list of predicted log probabilities
            log_probs.append(log_prob)

            # Update the previous token ID to be used as input for the next prediction
            prev_token_id = token_id

            # Compute the final log probability of the sentence by summing the predicted log probabilities
            log_prob_sentence = sum(log_probs)

            # Compute the final probability of the sentence
            prob_sentence = math.exp(log_prob_sentence)
            surp=-np.log(prob_sentence)
            val.append(surp)

            #val
            No = str(j+1) + '_' + str(i+1)
            sent_No.append(No)  
            df = pd.DataFrame(val, columns=['sent_surp'])
            df['sent_No'] = sent_No#str(j)+list(range(2, (len(sentences)+1)))
        
    appended_data.append(df)
appended_data = pd.concat(appended_data)


###############################################
##continue to compute sentence similarity
#############################################
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
dfy.to_csv('app_surp_sem1.csv')
