
##the python script provides how to use BERT 
#to compute sentence surprisal



#######################################################
#use BERT + chain rule to compute sentence surprisal
#################################################

#!/usr/bin/env python

import nltk
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM

# List of input files
filelist = ["Janus", "shaka", "doping", "thy", "worlden", "monocole", 
            "winetaste", "orangejuice", "beekeeping", "nationalflag", "union", "vr"]

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
model.eval()  # Set to evaluation mode
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

appended_data = []
for j, filename in enumerate(filelist):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    sentences = nltk.sent_tokenize(text)
    sentences = [s for s in sentences if s]  # Remove empty sentences

    val = []  # Store surprisal scores
    sent_No = []  # Store sentence identifiers

    for i in range(1, len(sentences)):  # Start from second sentence
        context = ' '.join(sentences[:i])  # All preceding sentences as context
        sentence = sentences[i]  # Current sentence

        # Tokenize the sentence
        tokens = tokenizer.tokenize(sentence)
        if not tokens:  # Skip empty sentences
            continue

        # Initialize log probability sum
        log_prob_sum = 0.0

        for k in range(len(tokens)):
            # Prepare input: context + tokens up to k, with k-th token masked
            input_tokens = tokenizer.tokenize(context) + tokens[:k] + [tokenizer.mask_token] + tokens[k+1:]
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            input_ids = torch.tensor([input_ids]).to(device)

            # Get model predictions
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits  # Shape: (1, seq_len, vocab_size)

            # Get probability of the true token at the masked position
            mask_pos = len(tokenizer.tokenize(context)) + k
            true_token_id = tokenizer.convert_tokens_to_ids([tokens[k]])[0]
            token_logits = logits[0, mask_pos, :]  # Logits for masked position
            probs = torch.softmax(token_logits, dim=-1)
            token_prob = probs[true_token_id].item()

            # Add log probability (avoid log(0))
            log_prob_sum += np.log(max(token_prob, 1e-10))

        # Compute surprisal: -log(P(S|C))
        surprisal = -log_prob_sum
        val.append(surprisal)

        # Store sentence identifier
        sent_No.append(f"{j+1}_{i+1}")  # Format: fileIndex_sentenceIndex

    # Create DataFrame for current file
    df = pd.DataFrame({'sent_surp': val, 'sent_No': sent_No})
    appended_data.append(df)

# Concatenate all DataFrames
appended_data = pd.concat(appended_data, ignore_index=True)

# Save to CSV
appended_data.to_csv('BERT_sentsurp.csv', index=False)








#########################################################################
#use "Next Sentence Prediction" in BERT to compute sentence surprisal

#############################################################

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
	
# Concatenate all DataFrames
appended_data = pd.concat(appended_data, ignore_index=True)

# Save to CSV
appended_data.to_csv('NSP_sentsurp.csv', index=False)
