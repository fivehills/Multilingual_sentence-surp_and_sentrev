################################################################
#use SentenceTransformer to compute sentence semantic relevance
###################################################################

import nltk
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# List of input files
filelist = [...]

# Weights for contextual sentences (c2, c1, cn1)
weights = [0.3, 0.5, 1.0]

# Initialize first model (paraphrase)
model_name_para = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_para = SentenceTransformer(model_name_para, device=device)

app_dat = []
for j, filename in enumerate(filelist):
    # Read and process the text file
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    sentences = nltk.sent_tokenize(text)
    sentences = [s for s in sentences if s]  # Remove empty sentences

    num = []  # Store sentrev scores
    sent_No = []  # Store sentence identifiers

    # Iterate through sentences to form sliding windows
    for i in range(2, len(sentences) - 1):  # Start at 2 to have sentence 2, end to have n1
        # Define the sliding window: sentence 2, 1, t, n1
        sent_2 = sentences[i - 2]  # Second preceding sentence
        sent_1 = sentences[i - 1]  # Immediate preceding sentence
        sent_t = sentences[i]      # Target sentence
        sent_n1 = sentences[i + 1] # Following sentence

        # Encode sentences to embeddings
        embeddings = model_para.encode([sent_2, sent_1, sent_t, sent_n1], 
                                      convert_to_tensor=True, device=device)
        emb_2, emb_1, emb_t, emb_n1 = embeddings

        # Compute cosine similarities between target and contextual sentences
        sim_2 = cosine_similarity(emb_t.cpu().numpy().reshape(1, -1), 
                                 emb_2.cpu().numpy().reshape(1, -1))[0][0]
        sim_1 = cosine_similarity(emb_t.cpu().numpy().reshape(1, -1), 
                                 emb_1.cpu().numpy().reshape(1, -1))[0][0]
        sim_n1 = cosine_similarity(emb_t.cpu().numpy().reshape(1, -1), 
                                  emb_n1.cpu().numpy().reshape(1, -1))[0][0]

        # Calculate sentrev as weighted sum
        sentrev = weights[0] * sim_2 + weights[1] * sim_1 + weights[2] * sim_n1

        # Store results
        num.append(sentrev)
        sent_No.append(f"{j+1}_{i+1}")  # Format: fileIndex_sentenceIndex

    # Create DataFrame for current file
    dx = pd.DataFrame({
        'sent_semrev_para': num,
        'sent_No': sent_No
    })
    app_dat.append(dx)

# Concatenate all DataFrames
app_dat1 = pd.concat(app_dat, ignore_index=True)

# Initialize second model (distiluse)
model_name_dist = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
model_dist = SentenceTransformer(model_name_dist, device=device)

new_dat = []
for j, filename in enumerate(filelist):
    # Read and process the text file
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    sentences = nltk.sent_tokenize(text)
    sentences = [s for s in sentences if s]  # Remove empty sentences

    num = []  # Store sentrev scores
    sent_No = []  # Store sentence identifiers

    # Iterate through sentences to form sliding windows
    for i in range(2, len(sentences) - 1):  # Start at 2 to have sentence 2, end to have n1
        # Define the sliding window: sentence 2, 1, t, n1
        sent_2 = sentences[i - 2]  # Second preceding sentence
        sent_1 = sentences[i - 1]  # Immediate preceding sentence
        sent_t = sentences[i]      # Target sentence
        sent_n1 = sentences[i + 1] # Following sentence

        # Encode sentences to embeddings
        embeddings = model_dist.encode([sent_2, sent_1, sent_t, sent_n1], 
                                      convert_to_tensor=True, device=device)
        emb_2, emb_1, emb_t, emb_n1 = embeddings

        # Compute cosine similarities between target and contextual sentences
        sim_2 = util.pytorch_cos_sim(emb_t, emb_2).item()
        sim_1 = util.pytorch_cos_sim(emb_t, emb_1).item()
        sim_n1 = util.pytorch_cos_sim(emb_t, emb_n1).item()

        # Calculate sentrev as weighted sum
        sentrev = weights[0] * sim_2 + weights[1] * sim_1 + weights[2] * sim_n1

        # Store results
        num.append(sentrev)
        sent_No.append(f"{j+1}_{i+1}")  # Format: fileIndex_sentenceIndex

    # Create DataFrame for current file
    dx = pd.DataFrame({
        'sent_semrev_dist': num,
        'sent_No': sent_No
    })
    new_dat.append(dx)

# Concatenate all DataFrames
new_dat1 = pd.concat(new_dat, ignore_index=True)

# Merge DataFrames
dfy = pd.merge(app_dat1, new_dat1, on='sent_No', how='left')

# Save to CSV
dfy.to_csv('sentTrans_sentrev.csv', index=False)
