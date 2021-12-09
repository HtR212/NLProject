from model import ClozeModel, read_cloze, torch, tqdm
import os

model = ClozeModel()
model.load_state_dict(torch.load("model-2-3000.pt"))
model.cuda()
model.eval()

corrects = 0
corrects_top2 = 0
total = 0


test_dir = "CLOTH/test/high"
for fname in tqdm(os.listdir(test_dir), desc="[Inferencing]"):
    if not fname.endswith(".json"):
        continue

    article_tokens, answers, option_ids = read_cloze(os.path.join(test_dir, fname))
    if article_tokens is None:
        continue

    with torch.no_grad():
        predictions = model(article_tokens.cuda(), option_ids)
        preds = torch.argmax(predictions, -1)
        corrects += (preds == answers.cuda()).sum()
        total += len(answers)

        tqdm.write(f"{corrects}/{total} = {corrects/total*100}%, {corrects_top2}/{total} = {corrects_top2/total*100}%")
