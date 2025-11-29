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
import gc
from tqdm import trange
import concurrent.futures
from typing import List, Optional
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords


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
        nlp = spacy.load("de_dep_news_trf", disable=["parser","ner"])
    except Exception as e:
        raise RuntimeError("Missing spacy German model. Run: python -m spacy download de_dep_news_trf") from e
    return nlp

def simple_preprocess(texts, nlp=None, batch_size=200):
    """
    Light preprocessing for German: lowercase, lemmatize, remove stopwords/punct/numbers.
    Returns cleaned texts (string) in a list.
    """
    if nlp is None:
        nlp = init_spacy()
    cleaned = []
    # process in batches to limit memory usage
    for i in trange(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        for doc in nlp.pipe(chunk, batch_size=32, n_process=1):
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

        # attempt to free intermediate objects after each batch
        try:
            gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return cleaned


def filter_out_region_ai(df, text_col='content', nlp=None, batch_size=100):
    """
    Examine texts in `df[text_col]` in batches and add a boolean column `filter_region`.
    `filter_region` == True means the document appears to be a regional use of "AI"
    (e.g., 'Kanton AI' referring to Appenzell Innerrhoden) and can be filtered out.

    This version processes rows in chunks using `nlp.pipe` and attempts to free
    memory after each batch (calls `gc.collect()` and `torch.cuda.empty_cache()`).
    """

    if nlp is None:
        nlp = init_spacy()

    flags = []
    texts = df[text_col].astype(str).tolist()

    re_flags = re.IGNORECASE

    for start in trange(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        # use nlp.pipe for efficient batch processing
        for doc in nlp.pipe(chunk, batch_size=32, n_process=1):
            text = doc.text.strip()
            is_region = False

            # quick check: if no 'AI' token, mark non-regional
            if not re.search(r'\bAI\b', text):
                flags.append(False)
                continue

            # 1) spaCy NER: check entities in the doc
            for ent in doc.ents:
                if ent.text.strip().upper() == "AI" and ent.label_ in ("LOC", "GPE", "ORG", "MISC"):
                    is_region = True
                    break
                if ent.label_ in ("LOC", "GPE") and re.search(r'Appenzell|Innerrhoden', ent.text, flags=re_flags):
                    is_region = True
                    break

            if is_region:
                flags.append(True)
                continue

            # 2) rule-based context
            region_terms = [
                r'\bKanton\b', r'\bKanton AI\b', r'\bAppenzell\b', r'\bAppenzell Innerrhoden\b',
                r'\bBezirk\b', r'\bGemeinde\b', r'\bKreis\b', r'\bStadt\b', r'\bKantonshauptort\b'
            ]
            region_pattern = re.compile("|".join(region_terms), flags=re_flags)
            if region_pattern.search(text) or re.search(r'\b(in|im|aus|aus dem|aus der|vom|von)\s+AI\b', text, flags=re_flags):
                flags.append(True)
                continue

            # 3) clearly mentions AI as 'künstliche Intelligenz' etc. -> non-regional
            if re.search(r'künstliche Intelligenz|künstliche-intelligenz|KI\b|artificial intelligence|AI-Systeme|AI-System', text, flags=re_flags):
                flags.append(False)
                continue

            # 4) windowed heuristics around 'AI' occurrences
            idxs = [m.start() for m in re.finditer(r'\bAI\b', text)]
            region_flag = False
            for idx in idxs:
                window = text[max(0, idx - 50): idx + 50]
                if re.search(r'Kanton|Gemeinde|Bezirk|Stadt|Kreis|Appenzell|Innerrhoden', window, flags=re_flags):
                    region_flag = True
                    break
                if re.search(r'künstliche|künstliche Intelligenz|KI|Maschine|Algorithmus|ML|Deep Learning', window, flags=re_flags):
                    region_flag = False
                    break

            flags.append(bool(region_flag))

        # attempt to free intermediate objects after each batch
        try:
            gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    df['filter_region'] = flags
    return df

# -------------------------
# Topic extraction (BERTopic)
# -------------------------
def extract_subtopics_bertopic(texts, embed_model, min_cluster_size=BERTOPIC_MIN_CLUSTER_SIZE, embed_batch_size=32, banned_words: Optional[List[str]] = None):
    """
    texts: list of raw (or preprocessed) strings
    embed_model: a sentence-transformers model instance (or model name)
    returns: topic model, topics, probs
    """
    # If embed_model is name, BERTopic will handle it; we prefer to pass embeddings to avoid double encode.
    # For simplicity encode here and pass to BERTopic.
    # prefer to accept a device string for GPU usage
    # embed_model may be a model name or already an instance
    # support multi-GPU encoding by passing a comma-separated devices string in embed_model (special case)
    # embed_model may be either a model name (str) or an instance. If embed_model is an instance
    # we cannot easily spawn processes that reuse it, so multi-GPU path requires a model name (str).
    if isinstance(embed_model, str):
        # use global torch device if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        s_model = SentenceTransformer(embed_model, device=device)
        print(f"Initialized embedding model '{embed_model}' on device: {device}")
    else:
        s_model = embed_model
        print (f"Using provided embedding model instance on device: {s_model.device}")

    # Encode embeddings in batches to reduce peak memory use.
    # If embed_model was supplied as a tuple (model_name, devices_list) we support multi-GPU
    # encoding by spawning worker processes, each loading the model on a different GPU.
    devices = None
    model_name_for_workers = None
    if isinstance(embed_model, tuple) and isinstance(embed_model[0], str):
        # embed_model passed as (model_name, devices_list)
        model_name_for_workers, devices = embed_model

    if devices and isinstance(devices, (list, tuple)) and len(devices) > 1 and model_name_for_workers is not None:
        # multi-GPU multiprocessing encoding
        def _worker_encode(args):
            # args = (model_name, device_id, texts_chunk)
            model_name, device_id, chunk_texts = args
            import os, gc
            import torch
            from sentence_transformers import SentenceTransformer
            # reduce tokenizer parallelism inside worker to avoid rayon issues
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            dev = f"cuda:{device_id}" if device_id != 'cpu' else 'cpu'
            m = SentenceTransformer(model_name, device=dev)
            emb = m.encode(chunk_texts, show_progress_bar=False, convert_to_numpy=True)
            try:
                del m
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return emb

        # split texts into N roughly-equal groups (per GPU) and within each group optionally further batch
        n_gpus = len(devices)
        # create chunks per gpu: simple round-robin or contiguous splits - use contiguous splits
        per_gpu = (len(texts) + n_gpus - 1) // n_gpus
        tasks = []
        for gi, dev in enumerate(devices):
            start = gi * per_gpu
            sub_texts = texts[start: start + per_gpu]
            if not sub_texts:
                continue
            # further split sub_texts into embed_batch_size pieces so worker encodes in smaller batches
            # Worker will load the model and encode its sub_texts in one call; to keep memory small,
            # we send smaller chunks and let the worker concatenate locally.
            # For simplicity, group sub_texts into one task per GPU (worker will handle batching internally)
            tasks.append((model_name_for_workers, dev, sub_texts))

        emb_parts = []
        # use ProcessPoolExecutor to avoid GIL and let each process use its GPU
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(tasks)) as ex:
            futures = [ex.submit(_worker_encode, t) for t in tasks]
            for f in concurrent.futures.as_completed(futures):
                emb_parts.append(f.result())
        # emb_parts order may not match original - reorder by task order
        # we used contiguous splits so we can sort by the start index in tasks
        # recreate embeddings in original order
        embeddings = np.vstack(emb_parts) if emb_parts else np.zeros((0, 0))

    else:
        # single-process encoding (existing logic)
        if embed_batch_size is None or embed_batch_size <= 0 or len(texts) <= embed_batch_size:
            embeddings = s_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        else:
            emb_chunks = []
            for i in trange(0, len(texts), embed_batch_size):
                chunk = texts[i:i+embed_batch_size]
                show = True if i == 0 else False
                emb = s_model.encode(chunk, show_progress_bar=show, convert_to_numpy=True)
                emb_chunks.append(emb)
                del emb
                gc.collect()
            embeddings = np.vstack(emb_chunks)

            # remove intermediate objects
            try:
                del emb_chunks
                gc.collect()
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
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
            print("Using CPU UMAP for dimensionality reduction")
        except Exception:
            umap_model = None
            print("UMAP not available; proceeding without dimensionality reduction")

    vectorizer_model = CountVectorizer(stop_words=banned_words) if banned_words is not None else None
    
    bm_kwargs = dict(language = "german", embedding_model=None, calculate_probabilities=False, verbose=False, min_topic_size=min_cluster_size)
    if umap_model is not None:
        bm_kwargs['umap_model'] = umap_model
    if vectorizer_model is not None:
        bm_kwargs['vectorizer_model'] = vectorizer_model
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
            # replace non case-sensitive
            new_t = re.sub(re.escape(kw), "", new_t, flags=re.IGNORECASE)
        cleaned.append(new_t)
    return cleaned

# -------------------------
# Main
# -------------------------
def main(args):
    # +) if banned words provided, load them
    banned_words = json.loads(args.banned)
    print(f"banned words: {banned_words} as type {type(banned_words)}")
    german_stopwords = stopwords.words('german')
    banned_words.extend(german_stopwords)

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
        df['combined_text'] = combined

        # 2) detect regional 'AI' mentions and mark them in a column
        df = filter_out_region_ai(df, text_col='combined_text', nlp=nlp)

        # 1) preprocess all docs (keep full alignment with df)
        cleaned_all = simple_preprocess(df['combined_text'].tolist(), nlp=nlp, batch_size=args.preprocess_batch_size)
        df['article_cleaned'] = cleaned_all

        # select non-regional docs for topic modeling
        mask_nonregional = ~df['filter_region']
        df = df.loc[mask_nonregional].copy()

    # 2-2) If already cleaned, just load cleaned
    else:
        if 'filter_region' in df.columns:
            mask_nonregional = ~df['filter_region']
            df = df.loc[mask_nonregional].copy()
        # keep alignment with df
        df['article_cleaned'] = df['article_cleaned'].astype(str).tolist()
        mask_nonregional = pd.Series([True] * len(df))
        

    # remove banned keywords from all cleaned texts
    texts_for_topics = remove_keywords(df['article_cleaned'].tolist(), banned_words)
    df['article_cleaned'] = texts_for_topics

    # save current df
    # temporal_output = args.input.rsplit('.',1)[0] + "_pre_topic_extraction.csv"
    # df.to_csv(temporal_output, index=False)
    # print(f"Saved pre-topic-extraction data to {temporal_output}")


    # Use cleaned texts (lemmatized, stopwords/punct/numbers removed) for topic extraction.
    print(f"Device for embeddings: {device}")
    print("Running BERTopic to extract subtopics (using cleaned texts)...")
    # instantiate sentence-transformers model on selected device to ensure GPU usage
    try:
        s_model = SentenceTransformer(args.embed_model, device=device)
    except Exception:
        # fallback: let extract_subtopics handle instantiation
        s_model = args.embed_model

    print("sample texts_for_topics (first 10):")
    for t in texts_for_topics[:10]:
        if any(bw in t for bw in banned_words):
            print(repr(t))

    topic_model, topics, probs = extract_subtopics_bertopic(texts_for_topics, embed_model=s_model,
                                                           min_cluster_size=args.min_cluster_size,
                                                           embed_batch_size=args.embed_batch_size,
                                                           banned_words = banned_words)

    # Optionally reduce/merge topics to a smaller number for coarser grouping
    # If the user provided --reduce_to (int), call BERTopic.reduce_topics.
    if getattr(args, 'reduce_to', None) is not None and int(args.reduce_to) > 0:
        try:
            target = int(args.reduce_to)
            print(f"Reducing topics to {target} using BERTopic.reduce_topics()")
            # reduce_topics returns (topic_model, topics, probs)
            topic_model, topics, probs = topic_model.reduce_topics(texts_for_topics, nr_topics=target)
            # update assignments in the original dataframe alignment
            df.loc[mask_nonregional, 'subtopic'] = topics
        except Exception as e:
            print("Topic reduction failed:", e)

    # create subtopic column aligned with original dataframe; non-regional rows receive topic assignments
    df['subtopic'] = -1
    df.loc[mask_nonregional, 'subtopic'] = topics

    # get topic info
    topic_info = topic_model.get_topic_info()  # DataFrame: Topic, Count, Name
    print("\nTop found topics (summary):")
    # print(topic_info.head(5).to_string(index=False))

    # Representative phrases for each topic (BERTopic gives top terms)
    topic_terms = {}
    for t in topic_info.Topic.unique():
        if t == -1:
            continue
        # get top 10 words for topic
        topic_terms[t] = [w for w, _ in topic_model.get_topic(t)]

    # attach results to edu_df (keep alignment with original df)
    edu_df = df[['head', 'subhead', 'pubtime', 'medium_name', 'article_cleaned', 'subtopic']].copy()
    # edu_raw_texts = df.get('combined_text', df['article_cleaned']).tolist()
    edu_df['subtopic_label'] = edu_df['subtopic'].apply(lambda t: topic_model.get_topic(t) if t >=0 else [])
    edu_df['subtopic_name'] = edu_df['subtopic'].apply(lambda t: topic_model.get_topic_info().loc[topic_model.get_topic_info().Topic==t,'Name'].values[0] if t in topic_model.get_topic_info().Topic.values else "outlier")

    # add YAKE keyphrases to help label subtopics (use raw texts to preserve phrases)
    # print("Extracting YAKE keyphrases for representative help...")
    # yake_phrases = extract_yake(edu_raw_texts, top=args.n_keyphrases, lang='de')
    # edu_df['yake_phrases'] = ["; ".join(k) for k in yake_phrases]

    # Save output
    outcols = ['id', 'head', 'subhead', 'pubtime', 'medium_name', 'article_cleaned','subtopic', 'subtopic_name']
    edu_df.to_csv(args.output, index=False, columns=[c for c in outcols if c in edu_df.columns])
    if not (args.cleaned):
        # save as input_cleaned.csv
        cleaned_output = args.input.rsplit('.',1)[0] + "_cleaned.csv"
        df.to_csv(cleaned_output, index=False)
        print(f"Saved cleaned data with regional AI filter to {cleaned_output}")
    # Attempt to free large objects and clear GPU memory
    try:
        del s_model
    except Exception:
        pass
    try:
        del topic_model
    except Exception:
        pass
    try:
        del topics
    except Exception:
        pass
    try:
        del probs
    except Exception:
        pass
    try:
        del texts_for_topics
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

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
    parser.add_argument("--reduce_to", type=int, default=None, help="If set, reduce topics to this number using BERTopic.reduce_topics")
    parser.add_argument("--preprocess_batch_size", type=int, default=200, help="Batch size for spaCy preprocessing (docs per batch)")
    parser.add_argument("--embed_batch_size", type=int, default=64, help="Batch size for embedding encoding (docs per batch)")
    args = parser.parse_args()
    main(args)