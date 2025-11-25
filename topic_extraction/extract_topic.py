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
import re
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import yake
from collections import Counter
import json
import torch

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

def init_spacy():
    # Default: load spaCy German model on CPU. If spaCy was configured for GPU
    # via `spacy.require_gpu()` before calling this, the model will be loaded on GPU.
    try:
        nlp = spacy.load("de_core_news_lg", disable=["parser","ner"])
    except Exception as e:
        raise RuntimeError("Missing spacy German model. Run: python -m spacy download de_core_news_lg") from e
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

def filter_out_region_ai(articles, nlp=None):
    '''
    Filter out regional "AI" articles (e.g., Kanton AI) from a list of texts.
    '''

    filtered = []

    if nlp is None:
        nlp = init_spacy()

    for doc_text in articles:
        text = doc_text.strip()
        # heuristic: if 'AI' token not present, likely not regional AI
        if not re.search(r'\bAI\b', text):
            filtered.append(text)
            continue

        # 1) spaCy NER for 'AI' labeled as location
        sp = nlp(text)
        is_region_by_ner = False
        for ent in sp.ents:
            # AI as GPE/LOC/ORG/MISC
            if ent.text.strip().upper() == "AI" and ent.label_ in ("LOC", "GPE", "ORG", "MISC"):
                is_region_by_ner = True
                break
            # entities like Appenzell labeled as LOC/GPE
            if ent.label_ in ("LOC", "GPE") and re.search(r'Appenzell|Innerrhoden', ent.text, flags=re.IGNORECASE):
                is_region_by_ner = True
                break

        if is_region_by_ner:
            continue

        # 2) Rule-based context check: 'Kanton AI', 'in AI', 'aus AI' etc.
        # example: "im Kanton AI", "aus AI", "in AI (Appenzell Innerrhoden)" etc.
        region_terms = [
        r'\bKanton\b', r'\bKanton AI\b', r'\bAppenzell\b', r'\bAppenzell Innerrhoden\b',
        r'\bBezirk\b', r'\bGemeinde\b', r'\bKreis\b', r'\bStadt\b', r'\bKantonshauptort\b'
    ]
        region_pattern = re.compile("|".join(region_terms), flags=re.IGNORECASE)
        if region_pattern.search(text) or re.search(r'\b(in|im|aus|aus dem|aus der|vom|von)\s+AI\b', text, flags=re.IGNORECASE):
            continue

        # 3) If AI keyword exists ('künstliche Intelligenz', 'KI' typical AI expressions) include
        if re.search(r'künstliche Intelligenz|künstliche-intelligenz|KI\b|artificial intelligence|AI-Systeme|AI-System', text, flags=re.IGNORECASE):
            filtered.append(text)
            continue

        # 4) Otherwise: if ambiguous, judge by document length and words around 'AI'
        idxs = [m.start() for m in re.finditer(r'\bAI\b', text)]
        region_flag = False
        for i in idxs:
            window = text[max(0, i-50): i+50]
            if re.search(r'Kanton|Gemeinde|Bezirk|Stadt|Kreis|Appenzell|Innerrhoden', window, flags=re.IGNORECASE):
                region_flag = True
                break
            if re.search(r'künstliche|künstliche Intelligenz|KI|Maschine|Algorithmen|ML|Deep Learning|ChatGPT|Large Language Model|LLM|Maschinelles Lernen|Machine Learning', window, flags=re.IGNORECASE):
                region_flag = False
                break
        if region_flag:
            continue

        filtered.append(text)

    return filtered

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
    # prefer to accept a device string for GPU usage
    # embed_model may be a model name or already an instance
    if isinstance(embed_model, str):
        # use global torch device if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        s_model = SentenceTransformer(embed_model, device=device)
    else:
        s_model = embed_model

    embeddings = s_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # Try to use GPU-accelerated UMAP from cuML (if available). Fall back to CPU UMAP.
    umap_model = None
    try:
        from cuml.manifold import UMAP as cuUMAP
        umap_model = cuUMAP(n_components=5)
        print("Using cuML UMAP (GPU) for dimensionality reduction")
    except Exception:
        try:
            from umap import UMAP
            umap_model = UMAP(n_components=5, random_state=42)
        except Exception:
            umap_model = None

    bm_kwargs = dict(embedding_model=None, calculate_probabilities=False, verbose=False, min_topic_size=min_cluster_size)
    if umap_model is not None:
        bm_kwargs['umap_model'] = umap_model

    topic_model = BERTopic(**bm_kwargs)
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

# remove certain keywords
def remove_keywords(texts, banned_keywords):
    cleaned = []
    for t in texts:
        new_t = t
        for kw in banned_keywords:
            new_t = new_t.replace(kw, "")
        cleaned.append(new_t)
    return cleaned

# -------------------------
# Main
# -------------------------
def main(args):
    # +) if banned words provided, load them
    banned_words = json.loads(args.banned)
    print(f"banned words: {banned_words}")

    # 1) Load data
    print("Loading data:", args.input)
    if args.input.endswith('.tsv'):
        df = pd.read_csv(args.input, sep='\t', on_bad_lines='skip')
    else: 
        df = pd.read_csv(args.input, on_bad_lines='skip')
    print(f"Loaded {len(df)} documents.")

    # 2-1) Preprocess texts (if not already cleaned)
    # determine device: prefer explicit CLI `--device`, else auto-detect
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Selected device: {device}")

    if not (args.cleaned):
        print("Initializing spaCy (German) and doing light preprocessing...")
        # if GPU requested, try to enable spaCy GPU
        if device == 'cuda':
            try:
                spacy.require_gpu()
                print("spaCy GPU enabled")
            except Exception as e:
                print("spaCy GPU requested but enable failed:", e)
                print("Falling back to CPU for spaCy")
        nlp = init_spacy()

        # parse html content if needed
        if 'content_parsed' not in df.columns:
            from bs4 import BeautifulSoup
            def parse_html(text):
                if pd.isna(text):
                    return ""
                return BeautifulSoup(text, 'html.parser').get_text(separator=' ')
            df['content_parsed'] = df['content'].apply(parse_html)

        # combine head/subhead/content_parsed if nan in head/subhead/content_parsed, convert to empty string      
        combined = (df['head'].astype(str) + ". " + df['subhead'].astype(str) + ". " + df['content_parsed'].astype(str)).tolist()
        AI_filtered = filter_out_region_ai(combined, nlp=nlp)
        cleaned = simple_preprocess(AI_filtered, nlp=nlp)

    # 2-2) If already cleaned, just load cleaned
    else:
        cleaned = df['article_cleaned'].tolist()
    
    cleaned = remove_keywords(cleaned, banned_words)
    df['article_cleaned'] = cleaned
    
    # Use cleaned texts (lemmatized, stopwords/punct/numbers removed) for topic extraction.
    print(f"Device for embeddings: {device}")
    print("Running BERTopic to extract subtopics (using cleaned texts)...")
    # instantiate sentence-transformers model on selected device to ensure GPU usage
    try:
        s_model = SentenceTransformer(args.embed_model, device=device)
    except Exception:
        # fallback: let extract_subtopics handle instantiation
        s_model = args.embed_model

    topic_model, topics, probs = extract_subtopics_bertopic(cleaned, embed_model=s_model,
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
    edu_df = df[['id', 'head', 'subhead', 'pubtime', 'medium_name', 'article_cleaned']].copy()
    edu_raw_texts = cleaned
    edu_df['subtopic'] = topics
    edu_df['subtopic_label'] = edu_df['subtopic'].apply(lambda t: topic_model.get_topic(t) if t >=0 else [])
    edu_df['subtopic_name'] = edu_df['subtopic'].apply(lambda t: topic_model.get_topic_info().loc[topic_model.get_topic_info().Topic==t,'Name'].values[0] if t in topic_model.get_topic_info().Topic.values else "outlier")

    # add YAKE keyphrases to help label subtopics (use raw texts to preserve phrases)
    print("Extracting YAKE keyphrases for representative help...")
    yake_phrases = extract_yake(edu_raw_texts, top=args.n_keyphrases, lang='de')
    edu_df['yake_phrases'] = ["; ".join(k) for k in yake_phrases]

    # Save output
    outcols = ['id', 'head', 'subhead', 'pubtime', 'medium_name', 'article_cleaned','subtopic', 'subtopic_name','yake_phrases']
    edu_df.to_csv(args.output, index=False, columns=[c for c in outcols if c in edu_df.columns])
    print(f"\nSaved education subtopic assignments to {args.output}")

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
    parser.add_argument("--cleaned", type=bool, default=False)
    parser.add_argument("--banned", type=str, default='[]')
    args = parser.parse_args()
    main(args)