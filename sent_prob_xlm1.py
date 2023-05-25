
###the following code for fintuing XLM-RoBERTa to compute sentence surprisal


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW

# Suppose that sentences and labels are stored in lists
sentences_1 = ["sentence_1_1", "sentence_2_1", ...]
sentences_2 = ["sentence_1_2", "sentence_2_2", ...]
labels = [0, 1, ...] 

tokenizer = RobertaTokenizer.from_pretrained('xlm-roberta-large')
model = RobertaForSequenceClassification.from_pretrained('xlm-roberta-large', num_labels=2)
model.to('cuda')
optimizer = AdamW(model.parameters(), lr=1e-5)

class NSPDataset(Dataset):
    def __init__(self, sent1, sent2, labels):
        self.sent1 = sent1
        self.sent2 = sent2
        self.labels = labels

    def __len__(self):
        return len(self.sent1)

    def __getitem__(self, idx):
        return self.sent1[idx], self.sent2[idx], self.labels[idx]

nsp_dataset = NSPDataset(sentences_1, sentences_2, labels)
nsp_dataloader = DataLoader(nsp_dataset, batch_size=32)

# Start training loop
for epoch in range(10): # Number of epochs, for example, 10 here
    for sent1, sent2, labels in nsp_dataloader:
        encoding = tokenizer(sent1, sent2, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to('cuda') for k, v in encoding.items()} 
        labels = labels.to('cuda') 

        outputs = model(**inputs, labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

