import re
import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from tqdm import tqdm
# import traceback

cuda = torch.device('cuda')

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.cuda()
model.eval()

corrects = 0
corrects_top2 = 0
total = 0

pattern = re.compile(r'([a-z ])([.?!])([A-Z])')

for i in tqdm(range(3172), desc="[Inferencing]"):
    try:
        with open(f"CLOTH/train/high/high{i}.json", "r") as f:
            data = json.load(f)

        # article = '[CLS] ' + data["article"]
        article = data["article"]
        article = pattern.sub(r'\1\2 \3', article)
        # article = re.sub(r'([a-z ])([.?!])( [A-Z])', r'\1\2 [SEP]\3', article)
        article = article.replace("_", "[MASK]")
        # article += '[SEP]'

        options = data["options"]
        idss = []
        for choices in options:
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

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(article))

    if len(indexed_tokens) > 512:
        continue

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens], device=cuda)
    segments_tensors = torch.zeros_like(tokens_tensor, device=cuda)

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors).logits
        masked_indices = torch.nonzero(tokens_tensor[0] == mask_id, as_tuple=True)[0]

        for answer, mid, ids in zip(answers, masked_indices, idss):
            probs = np.zeros(4, dtype=np.float32)
            for choice_id, tk_ids in enumerate(ids):
                probs[choice_id] = torch.max(predictions[0, mid, tk_ids])

            ranks = probs.argsort()
            corrects += ranks[-1] == answer
            corrects_top2 += (ranks[-1] == answer or ranks[-2] == answer)
            total += 1

        tqdm.write(f"{corrects}/{total} = {corrects/total*100}%, {corrects_top2}/{total} = {corrects_top2/total*100}%")
