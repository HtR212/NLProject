import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from tqdm import tqdm

cuda = torch.device('cuda')

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.cuda()
model.eval()

def calc_sent_prob(sent):
    batch_size = 32
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))
    token_ids = np.asarray(token_ids)
    token_batch = np.repeat(token_ids[np.newaxis], len(token_ids), 0)
    np.fill_diagonal(token_batch, mask_id)
    token_batch = torch.tensor(token_batch)
    segment_batch = torch.zeros_like(token_batch)

    log_prob = 0
    # create minibatches for long sequences
    for i in tqdm(range(0, len(token_ids), batch_size), desc="[Batch]"):
        stop = min(i + batch_size, len(token_ids))
        arange = np.arange(0, stop - i, dtype=np.int32)
        predictions = torch.nn.functional.log_softmax(
            model(token_batch[i:stop].cuda(), segment_batch[i:stop].cuda()).logits)
        log_prob += torch.sum(predictions[arange, arange, token_ids[i:stop]]).item()
    return log_prob
    

sent = "I sit outside of my daughter's nursery school classroom, patiently waiting for her. When the door opens, my daughter runs out with a broad smile. She   _   the excitement of school."
options = [
    "learns", 
    "adds", 
    "misses", 
    "enjoys"
]

for opt in options:
    temp = sent.replace("_", opt)
    print(calc_sent_prob(temp), temp)
