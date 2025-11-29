# 파일: filter_and_bertopic.py
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import spacy
import re

# 1) 로드: spaCy(독일어 NER), sentence-transformers 모델
nlp = spacy.load("de_core_news_lg")  # NER 성능 위해 large 권장
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def filter_out_region_ai(articles):
    filtered = []
    region_terms = [
        r'\bKanton\b', r'\bKanton AI\b', r'\bAppenzell\b', r'\bAppenzell Innerrhoden\b',
        r'\bBezirk\b', r'\bGemeinde\b', r'\bKreis\b', r'\bStadt\b', r'\bKantonshauptort\b'
    ]
    # 추가 키워드: 스위스 행정표현이나 'Kanton'이 근처에 있으면 지역 의미일 가능성 높음
    region_pattern = re.compile("|".join(region_terms), flags=re.IGNORECASE)

    for doc_text in articles:
        text = doc_text.strip()
        # 빠른 heuristic: 'AI' 토큰이 없으면 보통 인공지능 관련일 수도 있으니 포함
        if not re.search(r'\bAI\b', text):
            # 그래도 "künstliche Intelligenz" 등 포함 여부는 확인 안하면 BERTopic에서 잡힘.
            filtered.append(text)
            continue

        # 1) spaCy NER로 'AI'가 지명으로 라벨링 됐는지 확인
        sp = nlp(text)
        is_region_by_ner = False
        for ent in sp.ents:
            # ent.text가 'AI' (대소문자) 이고 라벨이 GPE/LOC 등일 경우
            if ent.text.strip().upper() == "AI" and ent.label_ in ("LOC", "GPE", "ORG", "MISC"):
                is_region_by_ner = True
                break
            # ent이 Appenzell 같은 지역명으로 잡혔을 경우도 지역 관련으로 처리
            if ent.label_ in ("LOC", "GPE") and re.search(r'Appenzell|Innerrhoden', ent.text, flags=re.IGNORECASE):
                is_region_by_ner = True
                break

        if is_region_by_ner:
            # 지역 'AI' 문서 -> 제외
            continue

        # 2) 규칙 기반 문맥검사: 'Kanton AI', 'in AI', 'aus AI' 등 지명표현
        # 예: "im Kanton AI", "aus AI", "in AI (Appenzell Innerrhoden)" 등
        if region_pattern.search(text) or re.search(r'\b(in|im|aus|aus dem|aus der|vom|von)\s+AI\b', text, flags=re.IGNORECASE):
            # 지역 의미일 가능성 높음 -> 제외
            continue

        # 3) 마지막 보정: 'künstliche Intelligenz', 'KI' 같은 인공지능 전형적 표현이 있으면 포함
        if re.search(r'künstliche Intelligenz|künstliche-intelligenz|KI\b|artificial intelligence|AI-Systeme|AI-System', text, flags=re.IGNORECASE):
            filtered.append(text)
            continue

        # 4) 그 외: 애매하면 문서 길이와 'AI' 주변 단어로 판정
        # 주변 20자 내 'künstliche' 같은 단어가 있으면 인공지능으로 간주
        idxs = [m.start() for m in re.finditer(r'\bAI\b', text)]
        region_flag = False
        for i in idxs:
            window = text[max(0, i-50): i+50]
            if re.search(r'Kanton|Gemeinde|Bezirk|Stadt|Kreis|Appenzell|Innerrhoden', window, flags=re.IGNORECASE):
                region_flag = True
                break
            if re.search(r'künstliche|künstliche Intelligenz|KI|Maschine|Algorithmus|ML|Deep Learning', window, flags=re.IGNORECASE):
                region_flag = False
                break
        if region_flag:
            continue

        # 기본적으로 포함
        filtered.append(text)

    return filtered

# 3) BERTopic 실행
def run_bertopic(docs, nr_topics=None):
    # docs: 독일어 문서 리스트(전처리 완료)
    embeddings = embed_model.encode(docs, show_progress_bar=True)
    topic_model = BERTopic(language="german", embedding_model=embed_model, calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(docs, embeddings)
    return topic_model, topics, probs

# 예시 사용법
if __name__ == "__main__":
    # articles = load_your_texts()  # 사용자가 이미 로드해 둔 독일어 기사 리스트
    articles = [
        "AI im Kanton Appenzell Innerrhoden meldet ...",   # 지역 -> 제외
        "Die Entwicklung von künstlicher Intelligenz (AI) in Deutschland ...", # 포함
        "Neue AI-Initiative der Stadt Bern ..."  # 지역인지 애매 -> 규칙에 따라 제거/보존
    ]
    filtered = filter_out_region_ai(articles)
    print("남은 문서 수:", len(filtered))
    model, topics, probs = run_bertopic(filtered)
    # 토픽 요약 출력
    print(model.get_topic_info().head(30))
