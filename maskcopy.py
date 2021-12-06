import re
import json
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from tqdm import tqdm
# import traceback

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
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
                choices = map(str.lower, choices)
                ids = tokenizer.convert_tokens_to_ids(choices)
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
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    masked_indices = []
    for i, token in enumerate(tokenized_text):
        if token == '[MASK]':
            masked_indices.append(i)

    predicted_indices = torch.argmax(predictions[0, np.array(masked_indices)[
                                     :, np.newaxis], np.array(idss)], axis=1).numpy()
    corrects += (predicted_indices == np.array(answers)).sum()
    total += len(answers)

    tqdm.write(f"{corrects}/{total} = {corrects/total*100}%")
    break
