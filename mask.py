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
total = 0
for i in tqdm(range(2341), desc="[Inferencing]"):
    try:
        with open(f"CLOTH/train/middle/middle{i}.json", "r") as f:
            data = json.load(f)

            article = data["article"]
            article = re.sub(r'([a-z ])([.?!])([A-Z])', r'\1\2 \3', article)
            article = article.replace("_", "[MASK]")

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
        max_prob = -1e10
        for choice_id, tk_ids in enumerate(ids):
            tk_prob = torch.max(predictions.logits[0, mid, tk_ids])
            if tk_prob > max_prob:
                max_prob = tk_prob
                best_choice = choice_id

        corrects += best_choice == answers[i]
        total += 1

    tqdm.write(f"{corrects}/{total} = {corrects/total*100}%")
