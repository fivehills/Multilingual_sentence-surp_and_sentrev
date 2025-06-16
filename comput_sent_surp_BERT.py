
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












