"""
education_subtopics_de.py

1) Loads a CSV (expects columns: 'title', 'content')
2) Filters articles related to 'education' (German) via keywords + semantic similarity
3) Runs BERTopic on the education subset to extract subtopics
4) Writes results to CSV: education_subtopics.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import yake
from collections import Counter

# -------------------------
# Config (tune these)
# -------------------------
DEFAULT_INPUT = "news.csv"              # input CSV
OUTPUT_CSV = "education_subtopics.csv"  # output for education subset with topic labels
EDU_KEYWORDS = [
    "bildung", "schule", "universität", "uni", "hochschule", "lehre", "erziehung",
    "schule", "kindergarten", "kita", "unterricht", "lehrer", "student", "studium",
    "pruefung", "prüfungen", "schulreform", "lern", "ausbildung", "pädagogik", "didaktik"
]
SEM_SIM_THRESHOLD = 0.62   # semantic similarity threshold to consider doc as education-related (0..1)
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # small & multilingual
MIN_EDU_DOCS = 10         # require at least this many docs to run BERTopic
BERTOPIC_MIN_CLUSTER_SIZE = 10  # HDBSCAN parameter (can be tuned)
N_KEYPHRASES = 6          # YAKE phrases per doc for extra labeling help

# -------------------------
# Utilities
# -------------------------
def load_data(path):
    df = pd.read_csv(path)
    if 'content' not in df.columns:
        for c in ['text','article','body', 'main_text']:
            if c in df.columns:
                df = df.rename(columns={c:'content'})
                break
    if 'content' not in df.columns:
        raise ValueError("Input CSV must contain a 'content' column (or rename yours).")
    if 'title' not in df.columns:
        df['title'] = ""
    df = df[['title','content']].fillna('')
    return df

def init_spacy():
    try:
        nlp = spacy.load("de_core_news_sm", disable=["parser","ner"])
    except Exception as e:
        raise RuntimeError("Missing spacy German model. Run: python -m spacy download de_core_news_sm") from e
    return nlp

def simple_preprocess(texts, nlp=None):
    """
    Light preprocessing for German: lowercase, lemmatize, remove stopwords/punct/numbers.
    Returns cleaned texts (string) in a list.
    """
    if nlp is None:
        nlp = init_spacy()
    cleaned = []
    for doc in nlp.pipe(texts, batch_size=50, n_process=1):
        toks = []
        for tok in doc:
            if tok.is_punct or tok.is_space or tok.is_stop:
                continue
            if tok.like_num:
                continue
            lemma = tok.lemma_.lower().strip()
            if len(lemma) < 2:
                continue
            toks.append(lemma)
        cleaned.append(" ".join(toks))
    return cleaned

# -------------------------
# Education filtering
# -------------------------
def filter_education(df, embed_model_name=EMBED_MODEL,
                     keywords=EDU_KEYWORDS, sem_sim_threshold=SEM_SIM_THRESHOLD):
    texts = (df['article_cleaned'].astype(str)).tolist()

    # 1) quick keyword match (case-insensitive)
    kw_mask = []
    lowered = [t.lower() for t in texts]
    for t in lowered:
        matched = any(kw in t for kw in keywords)
        kw_mask.append(matched)
    kw_mask = np.array(kw_mask)

    # 2) semantic similarity: compute embedding for each doc and for seed "education" vector
    #    This catches documents that discuss education without using exact keywords.
    model = SentenceTransformer(embed_model_name)
    doc_embeds = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # create a small set of German seed phrases representing "education"
    seed_phrases = [
        "Bildung", "Bildungspolitik", "Schulsystem", "Lehrerin", "Lehrer", "Schule", "Ausbildung", "Universität"
    ]
    seed_embeds = model.encode(seed_phrases, convert_to_numpy=True)
    seed_vec = seed_embeds.mean(axis=0)  # average seed embedding
    # cosine similarity function
    def cos_sim(a,b):
        a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
        return np.dot(a_norm, b_norm)

    sims = []
    # compute per-document cosine similarity to seed_vec
    seed_vec_norm = seed_vec / np.linalg.norm(seed_vec)
    for de in doc_embeds:
        sims.append(float(np.dot(de / np.linalg.norm(de), seed_vec_norm)))
    sims = np.array(sims)

    # Mask: either keyword matched OR sem sim above threshold
    edu_mask = (kw_mask) | (sims >= sem_sim_threshold)

    # Provide scores for sorting and debugging
    df2 = df.copy()
    df2['doc_text'] = texts
    df2['edu_keyword_match'] = kw_mask
    df2['edu_sem_sim'] = sims
    df2['edu_mask'] = edu_mask
    return df2, doc_embeds, model

# -------------------------
# Topic extraction (BERTopic)
# -------------------------
def extract_subtopics_bertopic(texts, embed_model, min_cluster_size=BERTOPIC_MIN_CLUSTER_SIZE):
    """
    texts: list of raw (or preprocessed) strings
    embed_model: a sentence-transformers model instance (or model name)
    returns: topic model, topics, probs
    """
    # If embed_model is name, BERTopic will handle it; we prefer to pass embeddings to avoid double encode.
    # For simplicity encode here and pass to BERTopic.
    if isinstance(embed_model, str):
        s_model = SentenceTransformer(embed_model)
    else:
        s_model = embed_model
    embeddings = s_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    topic_model = BERTopic(embedding_model=None,                   # we pass embeddings directly
                           calculate_probabilities=False,
                           verbose=False,
                           min_topic_size=min_cluster_size)
    topics, probs = topic_model.fit_transform(texts, embeddings)
    return topic_model, topics, probs

# -------------------------
# Keyphrase helper (YAKE)
# -------------------------
def extract_yake(texts, top=N_KEYPHRASES, lang='de'):
    kw = yake.KeywordExtractor(lan=lang, n=3, top=top)
    phrases = []
    for t in texts:
        k = kw.extract_keywords(t)
        phrases.append([p for p,s in k])
    return phrases

# -------------------------
# Main
# -------------------------
def main(args):
    print("Loading data:", args.input)
    df = load_data(args.input)
    print(f"Loaded {len(df)} documents.")

    print("Initializing spaCy (German) and doing light preprocessing...")
    nlp = init_spacy()
    combined = (df['title'].astype(str) + ". " + df['content'].astype(str)).tolist()
    cleaned = simple_preprocess(combined, nlp=nlp)
    # assign cleaned to a new column 'article_cleaned'
    df['article_cleaned'] = cleaned

    print("Filtering education-related documents (keywords + semantic similarity)...")
    df2, doc_embeds, s_model = filter_education(df, embed_model_name=args.embed_model,
                                               keywords=args.keywords, sem_sim_threshold=args.sem_sim_threshold)

    edu_df = df2[df2['edu_mask']].copy().reset_index(drop=True)
    # raw texts (keep for YAKE / output) and cleaned texts (for topic modeling)
    edu_raw_texts = (edu_df['title'].astype(str) + ". " + edu_df['content'].astype(str)).tolist()
    edu_cleaned_texts = edu_df['article_cleaned'].astype(str).tolist()
    print(f"Education-related documents found: {len(edu_df)} (out of {len(df)})")

    if len(edu_df) < args.min_edu_docs:
        print(f"Not enough education documents ({len(edu_df)}). Lower threshold or add more data.")
        # still save filtered docs so you can inspect
        edu_df.to_csv(args.output, index=False)
        return

    # Use cleaned texts (lemmatized, stopwords/punct/numbers removed) for topic extraction.
    print("Running BERTopic to extract subtopics (using cleaned texts)...")
    topic_model, topics, probs = extract_subtopics_bertopic(edu_cleaned_texts, embed_model=args.embed_model,
                                                           min_cluster_size=args.min_cluster_size)

    # get topic info
    topic_info = topic_model.get_topic_info()  # DataFrame: Topic, Count, Name
    print("\nTop found topics (summary):")
    print(topic_info.head(20).to_string(index=False))

    # Representative phrases for each topic (BERTopic gives top terms)
    topic_terms = {}
    for t in topic_info.Topic.unique():
        if t == -1:
            continue
        # get top 10 words for topic
        topic_terms[t] = [w for w, _ in topic_model.get_topic(t)]

    # attach results to edu_df
    edu_df['subtopic'] = topics
    edu_df['subtopic_label'] = edu_df['subtopic'].apply(lambda t: topic_model.get_topic(t) if t >=0 else [])
    edu_df['subtopic_name'] = edu_df['subtopic'].apply(lambda t: topic_model.get_topic_info().loc[topic_model.get_topic_info().Topic==t,'Name'].values[0] if t in topic_model.get_topic_info().Topic.values else "outlier")

    # add YAKE keyphrases to help label subtopics (use raw texts to preserve phrases)
    print("Extracting YAKE keyphrases for representative help...")
    yake_phrases = extract_yake(edu_raw_texts, top=args.n_keyphrases, lang='de')
    edu_df['yake_phrases'] = ["; ".join(k) for k in yake_phrases]

    # Save output
    outcols = ['title','content','edu_keyword_match','edu_sem_sim','subtopic','subtopic_name','yake_phrases']
    edu_df.to_csv(args.output, index=False, columns=[c for c in outcols if c in edu_df.columns])
    print(f"\nSaved education subtopic assignments to {args.output}")

    # Print short summary (top N docs per topic)
    print("\nRepresentative titles per subtopic:")
    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            continue
        mask = [t==topic_id for t in topics]
        indices = np.where(mask)[0]
        print(f"\nSubtopic {topic_id} (n={len(indices)}) top words:", topic_terms.get(topic_id, [])[:10])
        # print up to 3 representative titles
        for i in indices[:3]:
            print(" -", edu_df.loc[i,'title'][:180])
        # show common YAKE phrases across topic docs
        phrases = []
        for i in indices:
            phrases.extend(yake_phrases[i])
        most_common = Counter(phrases).most_common(5)
        print("  common phrases:", [p for p,c in most_common])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--embed_model", default=EMBED_MODEL)
    parser.add_argument("--sem_sim_threshold", type=float, default=SEM_SIM_THRESHOLD)
    parser.add_argument("--min_edu_docs", type=int, default=MIN_EDU_DOCS)
    parser.add_argument("--min_cluster_size", type=int, default=BERTOPIC_MIN_CLUSTER_SIZE)
    parser.add_argument("--keywords", nargs='*', default=EDU_KEYWORDS)
    parser.add_argument("--n_keyphrases", type=int, default=N_KEYPHRASES)
    args = parser.parse_args()
    main(args)
