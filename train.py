from transformers import AdamW
from model import ClozeModel, read_cloze, tqdm, torch, model_name
from transformers import get_scheduler
import pickle
import os
import random

cuda = torch.device('cuda')
model = ClozeModel()
model.to(cuda)
model.train()

optimizer = AdamW(model.parameters(), lr=5e-5)

if os.path.exists("train.pkl"):
    dataset = pickle.load(open("train.pkl", "rb"))
else:
    dataset = []
    for i in tqdm(range(3172), desc="[Inferencing]"):
        article_tokens, answers, option_ids = read_cloze(f"CLOTH/train/high/high{i}.json")
        if article_tokens is None:
            continue
        dataset.append((article_tokens, answers, option_ids))
    pickle.dump(dataset, open("train.pkl", "wb"))

num_epochs = 5
num_training_steps = num_epochs * len(dataset)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))
loss_func = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(num_epochs):
    corrects = 0
    total = 0
    random.shuffle(dataset)
    for i, (article_tokens, answers, option_ids) in enumerate(dataset):
        outputs = model(article_tokens.to(cuda), option_ids)
        answers = answers.to(cuda)
        corrects += torch.sum(answers == torch.argmax(outputs, -1)).item()
        total += len(answers)
        loss = loss_func(outputs, answers)
        if torch.isnan(loss):
            tqdm.write("Warning: nan loss")
            model.zero_grad()
            optimizer.zero_grad()
            lr_scheduler.step()
            continue

        tqdm.write(f"Accuracy: {corrects}/{total} = {corrects/total*100}% | Loss: {loss.item()}")

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        if i % 1000 == 0 and i > 0:
            torch.save(model.state_dict(), f"checkpoints/{model_name}-{epoch}-{i}.pt")