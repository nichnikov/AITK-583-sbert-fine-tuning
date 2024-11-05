import os
import re
from sentence_transformers import SentenceTransformer
from src.texts_processing import TextsTokenizer
from sentence_transformers.util import cos_sim
import pandas as pd


old_model_path = "/home/an/data/models/all_sys_paraphrase.transformers"
new_model_path = os.path.join(os.getcwd(), "models", "paraphrase_241105.transformers")

stopwords_df = pd.read_csv(os.path.join(os.getcwd(), "data", "stopwords.csv"), sep="\t")
greetings_df = pd.read_csv(os.path.join(os.getcwd(), "data", "greetings.csv"), sep="\t")
tokenizer = TextsTokenizer()
stopwords = greetings_df["stopwords"].tolist() + stopwords_df["stopwords"].tolist()
tokenizer.add_stopwords(stopwords)

pairs_df = pd.read_csv(os.path.join(os.getcwd(), "data", "val_dataset.csv"), sep="\t")
print(pairs_df)

queries = [" ".join(tkns) for tkns in tokenizer(pairs_df["query1"].tolist())]
etalons = [" ".join(tkns) for tkns in tokenizer(pairs_df["query2"].tolist())]

for model_path in [old_model_path, new_model_path]:
    model = SentenceTransformer(model_path)
    
    queries_embs = model.encode(queries)
    etalons_embs = model.encode(etalons)
    scores = []
    k = 1
    for qe, ee in zip(queries_embs, etalons_embs):
        print(k, "/", len(etalons))
        score = cos_sim(qe, ee)
        scores.append(score.item())
        k += 1 

    if model_path == old_model_path:
        scores_df = pd.DataFrame(scores, columns=["old_scores"])
    else:
        scores_df = pd.DataFrame(scores, columns=["new_scores"])
    pairs_df = pd.concat((pairs_df, scores_df), axis=1)

pairs_df.to_csv(os.path.join(os.getcwd(), "data", "old_new_compare2.csv"), sep="\t", index=False)