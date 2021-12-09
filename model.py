import torch
from transformers import BertTokenizer, BertForMaskedLM
import json
import numpy as np
import re
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

class ClozeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-large-uncased')

    def forward(self, token_ids, option_ids):
        attention_masks = torch.zeros_like(token_ids, device=token_ids.device)
        predictions = self.bert(token_ids, attention_masks).logits
        bidx, tidx = torch.nonzero(token_ids == mask_id, as_tuple=True)
        return predictions[bidx[:, None, None], tidx[:, None, None], option_ids].max(-1).values

pattern = re.compile(r'([a-z ])([.?!])([A-Z])')

def read_cloze(fpath):
    try:
        with open(fpath, "r") as f:
            data = json.load(f)

            # article = '[CLS] ' + data["article"]
            article = data["article"].replace("\n", " ")
            article = pattern.sub(r'\1\2 \3', article)
            # article = re.sub(r'([a-z ])([.?!])( [A-Z])', r'\1\2 [SEP]\3', article)
            article = article.replace("_", "[MASK]")
            # article += '[SEP]'

            options = data["options"]
            idss = []
            max_len = 0
            for choices in options:
                ids = []
                for choice in choices:
                    tids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(choice))
                    max_len = max(max_len, len(tids))
                    ids.append(tids)
                idss.append(ids)
            # make sure all the token ids are of the same length
            # idss shape: num blanks x 4 x max len
            for ids in idss:
                for tids in ids:
                    while len(tids) < max_len:
                        tids.append(tids[-1])
            if len(idss) == 0:
                tqdm.write("Skipping due to empty options")
                return None, None, None

            answers = [ord(answer) - 65 for answer in data["answers"]]
            article_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(article))

            if len(article_ids) > 512:
                return None, None, None

        return torch.tensor([article_ids]), torch.tensor(answers), torch.tensor(idss)
    except Exception as e:
        tqdm.write(str(e))
        return None, None, None