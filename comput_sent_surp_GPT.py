
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
       

def main():

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
