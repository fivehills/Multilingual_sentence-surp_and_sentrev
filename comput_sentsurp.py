
##the python script provides how to use BERT 
#to compute sentence surprisal



#######################################################
#use BERT + chain rule to compute sentence surprisal
#################################################
#!/usr/bin/env python

import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import nltk
import pandas as pd
import os
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load pre-trained mBERT model and tokenizer
print("Loading mBERT model and tokenizer...")
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

def compute_token_probability(input_ids, mask_idx, original_token_id):

    try:
        # Create a copy and mask the target position
        masked_ids = input_ids.copy()
        masked_ids[mask_idx] = tokenizer.mask_token_id
       
        # Convert to tensor and move to device
        input_tensor = torch.tensor([masked_ids], device=device)
       
        with torch.no_grad():
            outputs = model(input_tensor)
            predictions = outputs.logits[0, mask_idx]
       
        # Get probability of original token
        probs = torch.nn.functional.softmax(predictions, dim=0)
        original_token_prob = probs[original_token_id].item()
       
        # Ensure we don't return 0 probability
        return max(original_token_prob, 1e-10)
       
    except Exception as e:
        logger.warning(f"Error computing token probability at index {mask_idx}: {e}")
        return 1e-10

def compute_sentence_probability(context, sentence, max_length=512):

    try:
        # Tokenize context and sentence separately
        if context and context.strip():
            context_tokens = tokenizer.encode(context.strip(), add_special_tokens=False)
        else:
            context_tokens = []
           
        sentence_tokens = tokenizer.encode(sentence.strip(), add_special_tokens=False)
       
        if not sentence_tokens:  # Empty sentence
            return 1e-10
       
        # Construct full sequence: [CLS] context [SEP] sentence [SEP]
        if context_tokens:
            full_sequence = ([tokenizer.cls_token_id] +
                           context_tokens +
                           [tokenizer.sep_token_id] +
                           sentence_tokens +
                           [tokenizer.sep_token_id])
            sentence_start_idx = len(context_tokens) + 2  # +2 for [CLS] and [SEP]
        else:
            full_sequence = ([tokenizer.cls_token_id] +
                           sentence_tokens +
                           [tokenizer.sep_token_id])
            sentence_start_idx = 1  # +1 for [CLS]
       
        # Truncate if too long
        if len(full_sequence) > max_length:
            # Keep as much context as possible, but ensure sentence fits
            available_length = max_length - len(sentence_tokens) - 3  # -3 for [CLS], [SEP], [SEP]
            if available_length > 0 and context_tokens:
                context_tokens = context_tokens[-available_length:]
                full_sequence = ([tokenizer.cls_token_id] +
                               context_tokens +
                               [tokenizer.sep_token_id] +
                               sentence_tokens +
                               [tokenizer.sep_token_id])
                sentence_start_idx = len(context_tokens) + 2
            else:
                # Just use sentence if context is too long
                full_sequence = ([tokenizer.cls_token_id] +
                               sentence_tokens +
                               [tokenizer.sep_token_id])
                sentence_start_idx = 1
       
        sentence_end_idx = sentence_start_idx + len(sentence_tokens)
       
        # Compute probability for each token in the sentence
        sentence_log_prob = 0
       
        for i in range(sentence_start_idx, sentence_end_idx):
            if i >= len(full_sequence):
                break
               
            original_token_id = full_sequence[i]
            token_prob = compute_token_probability(full_sequence, i, original_token_id)
            sentence_log_prob += np.log(token_prob)
       
        # Return probability (not log probability)
        return np.exp(sentence_log_prob)
       
    except Exception as e:
        logger.warning(f"Error computing sentence probability: {e}")
        return 1e-10

def process_file(filename):
    """Process a single file and compute surprisal values."""
    try:
        if not os.path.exists(filename):
            logger.error(f"File {filename} not found")
            return None
           
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read().strip()
           
        if not text:
            logger.warning(f"File {filename} is empty")
            return None

        sentences = nltk.sent_tokenize(text)
       
        if not sentences:
            logger.warning(f"No sentences found in {filename}")
            return None
           
        print(f"Processing {filename} with {len(sentences)} sentences...")
       
        surprisals = []
       
        for i, sentence in enumerate(tqdm(sentences, desc=f"Processing {filename}")):
            if i == 0:  # First sentence has no context
                surprisals.append("NA")
            else:
                # Use all previous sentences as context
                left_context = ' '.join(sentences[:i])
                probability = compute_sentence_probability(left_context, sentence)
               
                # Compute surprisal (negative log probability)
                if probability > 0:
                    surprisal = -np.log(probability)
                else:
                    surprisal = float('inf')
               
                # Note: Not dividing by number of tokens as requested
                surprisals.append(surprisal)

        return surprisals, sentences
       
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        return None

def main():
    """Main function to process all files and save results."""
    filelist = [...]
   
    print(f"Current working directory: {os.getcwd()}")
   
    appended_data = []
   
    for idx, filename in enumerate(filelist, start=1):
        print(f"\n--- Processing file {idx}/{len(filelist)}: {filename} ---")
       
        result = process_file(filename)
        if result is None:
            continue
           
        surprisals, sentences = result
       
        # Create DataFrame - using total surprisal, not normalized by token count
        df = pd.DataFrame({
            'sent_surp_total': surprisals,  # Total surprisal (not averaged)
            'sent_No': [f"{idx}_{i}" for i in range(1, len(sentences) + 1)],
            'sentence': sentences,
            'file': filename,
            'sentence_length': [len(tokenizer.tokenize(s)) for s in sentences]  # For reference
        })
       
        appended_data.append(df)
       
        # Print statistics
        numeric_surprisals = [s for s in surprisals if s != "NA" and s != float('inf')]
        if numeric_surprisals:
            print(f"  Mean surprisal: {np.mean(numeric_surprisals):.2f}")
            print(f"  Std surprisal: {np.std(numeric_surprisals):.2f}")
            print(f"  Min surprisal: {np.min(numeric_surprisals):.2f}")
            print(f"  Max surprisal: {np.max(numeric_surprisals):.2f}")

    if appended_data:
        # Combine all data
        final_df = pd.concat(appended_data, ignore_index=True)
       
        # Save to CSV
        output_file = 'bert_surp_cr.csv'
        final_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        print(f"Total sentences processed: {len(final_df)}")
       
        # Print overall statistics
        numeric_surprisals = final_df[
            (final_df['sent_surp_total'] != "NA") &
            (final_df['sent_surp_total'] != float('inf'))
        ]['sent_surp_total'].astype(float)
       
        if len(numeric_surprisals) > 0:
            print(f"Overall mean surprisal: {numeric_surprisals.mean():.2f}")
            print(f"Overall std surprisal: {numeric_surprisals.std():.2f}")
           
        print(f"Sentences with infinite surprisal: {len(final_df[final_df['sent_surp_total'] == float('inf')])}")
       
if __name__ == "__main__":
    main()









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


#########################################################################
#use "NLL" in GPT to compute sentence surprisal

#############################################################

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import nltk
import pandas as pd
from tqdm import tqdm
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load pre-trained mGPT model and tokenizer
try:
    print("Loading mGPT model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('ai-forever/mGPT')
    model = GPT2LMHeadModel.from_pretrained('ai-forever/mGPT')
    model.eval()
   
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
       
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

def compute_sentence_nll(context, sentence, max_context_length=1024):
 
    try:
        # Tokenize context and sentence separately
        if context and context.strip():
            context_tokens = tokenizer.encode(context.strip(), add_special_tokens=False)
            # Truncate context if too long
            if len(context_tokens) > max_context_length:
                context_tokens = context_tokens[-max_context_length:]
        else:
            context_tokens = []
           
        sentence_tokens = tokenizer.encode(sentence.strip(), add_special_tokens=False)
       
        if not sentence_tokens:  # Empty sentence
            return float('inf')
       
        # Combine tokens
        all_tokens = context_tokens + sentence_tokens
        input_ids = torch.tensor([all_tokens], device=device)
       
        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
       
        # Calculate cross-entropy loss for sentence tokens only
        sentence_start_idx = len(context_tokens)
        sentence_end_idx = len(all_tokens)
       
        if sentence_start_idx >= sentence_end_idx:
            return float('inf')
       
        target_logits = logits[0, sentence_start_idx-1:sentence_end_idx-1, :]
        target_tokens = torch.tensor(sentence_tokens, device=device)
       
        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        nll = loss_fct(target_logits, target_tokens).item()
       
        return nll
       
    except Exception as e:
        logger.warning(f"Error computing NLL: {e}")
        return float('inf')

def process_file(filename, max_context_length=1024, context_mode='cumulative'):

    try:
        if not os.path.exists(filename):
            logger.error(f"File not found: {filename}")
            return None
           
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read().strip()
           
        if not text:
            logger.warning(f"File {filename} is empty")
            return None

        sentences = nltk.sent_tokenize(text)
       
        if not sentences:
            logger.warning(f"No sentences found in {filename}")
            return None
           
        print(f"Processing {filename} with {len(sentences)} sentences...")
       
        nll_values = []
        perplexity_values = []
        context_lengths = []
       
        for i, sentence in enumerate(tqdm(sentences, desc=f"Processing {filename}", leave=False)):
            if i == 0:  # First sentence has no context
                nll_values.append("NA")

            else:
                if context_mode == 'cumulative':
                    # Use all previous sentences as context
                    context = ' '.join(sentences[:i])
                elif context_mode == 'sliding':
                    # Use only recent sentences (sliding window)
                    window_size = 5  # Adjust as needed
                    start_idx = max(0, i - window_size)
                    context = ' '.join(sentences[start_idx:i])
                else:
                    raise ValueError(f"Unknown context_mode: {context_mode}")
               
                # Compute NLL and perplexity
                nll = compute_sentence_nll(context, sentence, max_context_length)
               
                nll_values.append(nll)

        return {
            'sentences': sentences,
            'nll_values': nll_values
        }
       
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        return None

def main():
    """Main function to process all files and save results."""
    filelist = [...]
   
    print(f"Current working directory: {os.getcwd()}")
   
    appended_data = []
   
    for idx, filename in enumerate(tqdm(filelist, desc="Processing files"), start=1):
        print(f"\n--- Processing file {idx}/{len(filelist)}: {filename} ---")
       
        result = process_file(filename, max_context_length=1024, context_mode='cumulative')
        if result is None:
            continue
           
        sentences = result['sentences']
        nll_values = result['nll_values']
        perplexity_values = result['perplexity_values']
        context_lengths = result['context_lengths']
       
        # Create DataFrame
        df = pd.DataFrame({
            'gpt_nll_sp': nll_values,
            'sent_No': [f"{idx}_{i}" for i in range(1, len(sentences) + 1)],
            'sentence': sentences,
            'file': filename
        })
       
        appended_data.append(df)
       
        # Print statistics
        numeric_nll = [x for x in nll_values if x != "NA" and x != float('inf') and not np.isnan(x)]
        if numeric_nll:
            print(f"  Mean NLL: {np.mean(numeric_nll):.2f}")
            print(f"  Std NLL: {np.std(numeric_nll):.2f}")
            print(f"  Min NLL: {np.min(numeric_nll):.2f}")
            print(f"  Max NLL: {np.max(numeric_nll):.2f}")
           
        numeric_ppl = [x for x in perplexity_values if x != "NA" and x != float('inf') and not np.isnan(x)]
        if numeric_ppl:
            print(f"  Mean Perplexity: {np.mean(numeric_ppl):.2f}")

    if appended_data:
        # Combine all data
        final_df = pd.concat(appended_data, ignore_index=True)
       
        # Save to CSV
        output_file = 'GPT_nll_surp_improved.csv'
        final_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        print(f"Total sentences processed: {len(final_df)}")
       
        # Print overall statistics
        numeric_nll = final_df[
            (final_df['gpt_nll_sp'] != "NA") &
            (final_df['gpt_nll_sp'] != float('inf'))
        ]['gpt_nll_sp']
       
        if len(numeric_nll) > 0:
            numeric_nll = numeric_nll.astype(float)
            print(f"Overall mean NLL: {numeric_nll.mean():.2f}")
            print(f"Overall std NLL: {numeric_nll.std():.2f}")
           
        print(f"Sentences with infinite NLL: {len(final_df[final_df['gpt_nll_sp'] == float('inf')])}")
    else:
        print("No data processed. Check if files exist and are accessible.")

if __name__ == "__main__":
    main()
