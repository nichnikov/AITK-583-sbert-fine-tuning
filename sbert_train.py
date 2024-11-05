import os
import re
import pandas as pd
from random import shuffle
from sentence_transformers import (SentenceTransformer,
                                   losses,
                                   InputExample)
from sentence_transformers import evaluation
from torch.utils.data import DataLoader
from src.texts_processing import TextsTokenizer
import optuna
import math


def callback(value, a, b):
    print('\ncallback:', "eval:", value, "epoch:", a, "step:", b)
    if math.isnan(value):
        raise optuna.exceptions.TrialPruned()

# model_path = "/home/an/data/models/all_sys_paraphrase.transformers"
model_path = os.path.join("models", "paraphrase_241105.transformers")
model = SentenceTransformer(model_path)


tokenizer = TextsTokenizer()
stopwords_df = pd.read_csv(os.path.join(os.getcwd(), "data", "stopwords.csv"), sep="\t")
greetings_df = pd.read_csv(os.path.join(os.getcwd(), "data", "greetings.csv"), sep="\t")
tokenizer = TextsTokenizer()
stopwords = greetings_df["stopwords"].tolist() + stopwords_df["stopwords"].tolist()
tokenizer.add_stopwords(stopwords)


pairs_df = pd.read_csv(os.path.join(os.getcwd(), "data", "train_pairs.csv"), sep="\t")
queries_clear = [re.sub("\n+", " ", tx) for tx in pairs_df["query"].tolist()]
queries_lem =[" ".join(tkns) for tkns in tokenizer(queries_clear)]
etalons_lem =[" ".join(tkns) for tkns in tokenizer(pairs_df["etalon"].tolist())]
scores = [float(lb) for lb in pairs_df["label"].tolist()]

new_dataset = [(lm_q1, lm_q2, sc) for lm_q1, lm_q2, sc in zip(queries_lem, etalons_lem, scores)]
shuffle(new_dataset)
print(new_dataset[:10])



train_dataset_df = pd.read_csv(os.path.join("data", "train_dataset.csv"), sep="\t")
val_dataset_df = pd.read_csv(os.path.join("data", "val_dataset.csv"), sep="\t")


train_dataset = list(train_dataset_df[:1000].itertuples(index=False, name=None)) + new_dataset
shuffle(train_dataset)
print(train_dataset[:10])
print("len train_dataset:", len(train_dataset))

train_examples = [InputExample(texts=[tx1, tx2], label=sc) for tx1, tx2, sc in train_dataset]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=40)
train_loss = losses.CosineSimilarityLoss(model)

val_dataset = list(val_dataset_df[:200].itertuples(index=False, name=None))
# val_dataset = train_dataset 
sentences1, sentences2, scores = zip(*val_dataset)
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

# Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          evaluation_steps=50,
          epochs=3,
          warmup_steps=3,
          # checkpoint_save_steps=1000,
          # checkpoint_path=os.path.join("models", "paraphrase_241105.transformers"),
          callback=callback,
          )

model.save(os.path.join("models", "paraphrase_241105.transformers"))