import torch
from model import ClozeModel, read_cloze
from tqdm import tqdm

model = ClozeModel()
model.cuda()
model.eval()

corrects = 0
corrects_top2 = 0
total = 0


for i in tqdm(range(3172), desc="[Inferencing]"):
    article_tokens, answers, option_ids = read_cloze(f"CLOTH/train/high/high{i}.json")
    if article_tokens is None:
        continue

    with torch.no_grad():
        predictions = model(article_tokens.cuda(), option_ids)
        preds = torch.argmax(predictions, -1)
        corrects += (preds.cpu() == answers).sum()
        total += len(answers)

        tqdm.write(f"{corrects}/{total} = {corrects/total*100}%, {corrects_top2}/{total} = {corrects_top2/total*100}%")
