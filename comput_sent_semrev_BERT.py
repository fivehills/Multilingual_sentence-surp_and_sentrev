####the python script is for computing sentence relevance

###the following script used "BERT" to generate sentence embeddings respectively.


############################
#use GPT to compute
############################

#!/usr/bin/env python



import nltk
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer(model_name, device=device)

# List of input files
filelist = ["Janus", "shaka", "doping", "thy", "worlden", "monocole", 
            "winetaste", "orangejuice", "beekeeping", "nationalflag", "union", "vr"]

# Weights for contextual sentences (c2, c1, cn1)
weights = [0.3, 0.5, 1.0]

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
        embeddings = model.encode([sent_2, sent_1, sent_t, sent_n1], 
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
        'sent_semrev': num,
        'sent_No': sent_No
    })
    app_dat.append(dx)

# Concatenate all DataFrames
app_dat1 = pd.concat(app_dat, ignore_index=True)

# Save to CSV
app_dat1.to_csv('GPT_sentrev.csv', index=False)




##################################
#Use the BERT to compute
###################################


#!/usr/bin/env python

import nltk
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

# List of input files
filelist = [...]

# Weights for contextual sentences (c2, c1, cn1)
weights = [0.3, 0.5, 1.0]

# Initialize BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased")
cosine_similarity = torch.nn.CosineSimilarity(dim=1)

app_dat = []
for j, filename in enumerate(filelist):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    sentences = nltk.sent_tokenize(text)
    sentences = [s for s in sentences if s]  # Remove empty sentences

    num = []  # Store sentrev scores
    sent_No = []

    # Iterate through sentences to form sliding windows
    for i in range(2, len(sentences) - 1):  # Start at 2 to have sentence 2, end to have n1
        # Define the sliding window: sentence 2, 1, t, n1
        sent_2 = sentences[i - 2]  # Second preceding sentence
        sent_1 = sentences[i - 1]  # Immediate preceding sentence
        sent_t = sentences[i]      # Target sentence
        sent_n1 = sentences[i + 1] # Following sentence

        # Tokenize and encode sentences
        inputs = tokenizer([sent_2, sent_1, sent_t, sent_n1], return_tensors='pt', 
                          padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.pooler_output  # Shape: (4, 768)
        emb_2, emb_1, emb_t, emb_n1 = embeddings

        # Compute cosine similarities between target and contextual sentences
        sim_2 = cosine_similarity(emb_t.unsqueeze(0), emb_2.unsqueeze(0)).item()
        sim_1 = cosine_similarity(emb_t.unsqueeze(0), emb_1.unsqueeze(0)).item()
        sim_n1 = cosine_similarity(emb_t.unsqueeze(0), emb_n1.unsqueeze(0)).item()

        # Calculate sentrev as weighted sum
        sentrev = weights[0] * sim_2 + weights[1] * sim_1 + weights[2] * sim_n1

        # Store results
        num.append(sentrev)
        sent_No.append(f"{j+1}_{i+1}")  # Format: fileIndex_sentenceIndex

    # Create DataFrame
    dx = pd.DataFrame({'sent_semrev': num, 'sent_No': sent_No})
    app_dat.append(dx)

# Concatenate all DataFrames
app_dat1 = pd.concat(app_dat, ignore_index=True)

# Save to CSV
app_dat1.to_csv('BERT_sentrev.csv', index=False)


