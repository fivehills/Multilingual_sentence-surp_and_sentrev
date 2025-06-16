
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
