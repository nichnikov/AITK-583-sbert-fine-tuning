import os
import re
from sentence_transformers import SentenceTransformer
from src.texts_processing import TextsTokenizer
from sentence_transformers.util import cos_sim
import pandas as pd



# model_path = "/home/an/data/models/all_sys_paraphrase.transformers"
model_path = os.path.join(os.getcwd(), "models", "paraphrase_241105.transformers")
model = SentenceTransformer(model_path)

stopwords_df = pd.read_csv(os.path.join(os.getcwd(), "data", "stopwords.csv"), sep="\t")
greetings_df = pd.read_csv(os.path.join(os.getcwd(), "data", "greetings.csv"), sep="\t")
tokenizer = TextsTokenizer()
stopwords = greetings_df["stopwords"].tolist() + stopwords_df["stopwords"].tolist()
tokenizer.add_stopwords(stopwords)

pairs_df = pd.read_csv(os.path.join(os.getcwd(), "data", "train_pairs_with_scores.csv"), sep="\t")
print(pairs_df)

queries_clear = [re.sub("\n+", " ", tx) for tx in pairs_df["query"].tolist()]
queries =[" ".join(tkns) for tkns in tokenizer(queries_clear)]
etalons =[" ".join(tkns) for tkns in tokenizer(pairs_df["etalon"].tolist())]
queries_embs = model.encode(queries)
etalons_embs = model.encode(etalons)

scores = []
for qe, ee in zip(queries_embs, etalons_embs):
    score = cos_sim(qe, ee)
    scores.append(score.item())

scores_df = pd.DataFrame(scores, columns=["new_scores"])
pairs_with_scores_df = pd.concat((pairs_df, scores_df), axis=1)
# print(pairs_with_scores_df[pairs_with_scores_df["scores"] <= 0.5])
pairs_with_scores_df.to_csv(os.path.join(os.getcwd(), "data", "train_pairs_with_scores.csv"), sep="\t", index=False)