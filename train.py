from transformers import AdamW
from model import ClozeModel, read_cloze, tqdm, torch
from transformers import get_scheduler
import pickle
import os

cuda = torch.device('cuda')
model = ClozeModel()
model.cuda()
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

num_epochs = 3
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
    for article_tokens, answers, option_ids in dataset:
        outputs = model(article_tokens.cuda(), option_ids)

        loss = loss_func(outputs, answers.cuda())
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)