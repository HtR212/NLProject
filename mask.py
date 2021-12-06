import re
import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from tqdm import tqdm
# import traceback

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.cuda()
model.eval()

corrects = 0
corrects_top2 = 0
total = 0
for i in tqdm(range(3172), desc="[Inferencing]"):
    try:
        with open(f"CLOTH/train/high/high{i}.json", "r") as f:
            data = json.load(f)

            # article = '[CLS] ' + data["article"]
            article = data["article"]
            article = re.sub(r'([a-z ])([.?!])([A-Z])', r'\1\2 \3', article)
            # article = re.sub(r'([a-z ])([.?!])( [A-Z])', r'\1\2 [SEP]\3', article)
            article = article.replace("_", "[MASK]")
            # article += '[SEP]'

            options = data["options"]
            idss = []
            for choices in options:
                # tokenizer.convert_tokens_to_ids(map(str.lower, choices))
                ids = [
                    tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(choice)) for choice in choices
                ]
                idss.append(ids)

            answers = data["answers"]
            answers = [ord(answer) - 65 for answer in answers]
    except Exception as e:
        print(e)
        continue

    text = article

    # Load pre-trained model tokenizer (vocabulary)

    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    if len(indexed_tokens) > 512:
        continue

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).cuda()
    segments_tensors = torch.tensor([segments_ids]).cuda()

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    masked_indices = []
    for i, token in enumerate(tokenized_text):
        if token == '[MASK]':
            masked_indices.append(i)

    for i, (mid, ids) in enumerate(zip(masked_indices, idss)):
        probs = np.zeros(4, dtype=np.float32)
        for choice_id, tk_ids in enumerate(ids):
            probs[choice_id] = torch.max(predictions.logits[0, mid, tk_ids])

        ranks = probs.argsort()
        corrects += ranks[-1] == answers[i]
        corrects_top2 += (ranks[-1] == answers[i] or ranks[-2] == answers[i])
        total += 1

    tqdm.write(f"{corrects}/{total} = {corrects/total*100}%, {corrects_top2}/{total} = {corrects_top2/total*100}%")
