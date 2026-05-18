from flask import Flask, send_from_directory, jsonify, request
import json
import os
from datetime import datetime, timezone

app = Flask(__name__, static_folder='frontend')

# ── File paths ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DOCS_DATA_DIR = os.path.join(os.path.dirname(__file__), 'docs', 'data')

DATA_FILES = {
    'de': 'run_swissdocx_0322_Germany_20260413_214420.json',
    'fr': 'run_swissdocx_0322_French_20260413_192120.json',
    'it': 'run_swissdocx_0322_Italian_20260413_183132.json',
}

LANG_LABELS = {
    'de': 'Deutsch',
    'fr': 'Français',
    'it': 'Italiano',
}

LANG_COLORS = {
    'de': '#b5122a',
    'fr': '#1a5fa8',
    'it': '#1a8a3a',
}

# ── Source metadata ─────────────────────────────────────────────────────────
SOURCE_NAMES = {
    'NZZ': 'NZZ', 'NZZM': 'NZZ am Sonntag', 'NZZO': 'NZZ Online',
    'TA': 'Tages-Anzeiger', 'TAM': 'Tages-Anzeiger', 'TASI': 'Tages-Anzeiger',
    'SRF': 'SRF', 'SRFA': 'SRF', 'SRFV': 'SRF', 'RSI': 'RSI',
    'RTS': 'RTS', 'RTR': 'RTR',
    'BLI': 'Blick', 'BLIO': 'Blick Online',
    'AZ': 'Aargauer Zeitung', 'AZO': 'Aargauer Zeitung Online',
    'BZ': 'Basler Zeitung', 'BAZ': 'Basler Zeitung',
    'BEOL': 'Berner Zeitung', 'BEOLO': 'Berner Zeitung Online',
    'WOZ': 'WOZ', 'SF': 'Sonntagszeitung', 'SONT': 'Sonntagszeitung',
    'CASH': 'Cash', 'CASO': 'Cash Online',
    'FUW': 'Finanz und Wirtschaft', 'FUWO': 'Finanz und Wirtschaft Online',
    'SGT': 'St. Galler Tagblatt', 'SGTO': 'St. Galler Tagblatt Online',
    'LUZ': 'Luzerner Zeitung', 'SHZ': 'Schweizer Handelszeitung',
    'SHZO': 'Schweizer Handelszeitung Online',
    'ZWAO': 'Zentralschweiz Online', 'NNBS': 'Basler Zeitung',
    'NNTA': 'Tages-Anzeiger',
    'LT': 'Le Temps', 'LTO': 'Le Temps Online',
    '24H': '24 Heures', '24HO': '24 Heures Online',
    'TDG': 'Tribune de Genève', 'TDGO': 'Tribune de Genève Online',
    'LE': "L'Express", 'LI': "L'Impartial",
    'COR': 'Le Courrier', 'CORO': 'Le Courrier Online',
}

SOURCE_COLORS = {
    'NZZ': '#7a6ab8', 'NZZM': '#7a6ab8', 'NZZO': '#7a6ab8',
    'TA': '#3a5a8a', 'TAM': '#3a5a8a', 'NNTA': '#3a5a8a',
    'SRF': '#c06030', 'SRFA': '#c06030', 'SRFV': '#c06030',
    'RSI': '#1a8a3a', 'RTS': '#1a5fa8',
    'BLI': '#D45B5B', 'BLIO': '#D45B5B',
    'CASO': '#B5122A', 'CASH': '#B5122A',
    'NZZO': '#4a4844', 'FUWO': '#c09030', 'FUW': '#c09030',
    'SGTO': '#2a7a6a', 'SGT': '#2a7a6a',
    'SHZO': '#3a6a9a', 'SHZ': '#3a6a9a',
    'ZWAO': '#8a6a2a', 'NNBS': '#7a4a8a',
    'LT': '#1a5fa8', 'LTO': '#1a5fa8',
    '24H': '#2a7a9a', '24HO': '#2a7a9a',
    'TDG': '#5a3a8a', 'TDGO': '#5a3a8a',
}


def get_source_display(code):
    return SOURCE_NAMES.get(code, code)

def get_source_color(code):
    return SOURCE_COLORS.get(code, '#7A7A8A')


# ── Load topic mapping from preproc.json ──────────────────────────────────
TOPIC_MAPPING = {}
SENTIMENT_MAPPING = {}
ITEMS_ARTICLES = []

def load_topic_mapping():
    global TOPIC_MAPPING, SENTIMENT_MAPPING, ITEMS_ARTICLES
    items_index_path = os.path.join(DOCS_DATA_DIR, 'items_index.json')
    if not os.path.exists(items_index_path):
        print('WARNING: items_index.json not found')
        return
    with open(items_index_path, 'r', encoding='utf-8') as f:
        items_index = json.load(f)

    SOURCE_NAME_MAP = {
        'NZZ':'NZZ','NZZM':'NZZ am Sonntag','NZZO':'NZZ Online',
        'TA':'Tages-Anzeiger','SRF':'SRF','RSI':'RSI','RTS':'RTS',
        'BLI':'Blick','BLIO':'Blick Online','AZ':'Aargauer Zeitung',
        'BAZ':'Basler Zeitung','BEOL':'Berner Zeitung','WOZ':'WOZ',
        'FUW':'Finanz und Wirtschaft','FUWO':'Finanz und Wirtschaft Online',
        'SGT':'St. Galler Tagblatt','LUZ':'Luzerner Zeitung',
        'LT':'Le Temps','LTO':'Le Temps Online',
        '24H':'24 Heures','TDG':'Tribune de Genève',
        'TPS':'TPS','TPSO':'TPS Online','ZWSO':'Zentralschweiz Online',
        'ZWA':'Zentralschweiz','NNHEU':'Neue Helvetische Gesellschaft',
    }
    SOURCE_COLOR_MAP = {
        'NZZ':'#7a6ab8','SRF':'#c06030','RSI':'#1a8a3a','RTS':'#1a5fa8',
        'BLI':'#D45B5B','BLIO':'#D45B5B','TA':'#3a5a8a',
        'LT':'#1a5fa8','LTO':'#1a5fa8','24H':'#2a7a9a','TDG':'#5a3a8a',
        'TPS':'#4a8a7a','TPSO':'#4a8a7a','ZWA':'#8a6a2a','ZWSO':'#8a6a2a',
    }

    for broad_topic, url in items_index.items():
        filename = os.path.basename(url)
        items_path = os.path.join(DOCS_DATA_DIR, filename)
        if not os.path.exists(items_path):
            continue
        with open(items_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
        for item in items:
            if not isinstance(item, list) or len(item) < 8:
                continue
            article_id = str(item[0])
            pubtime    = str(item[1] or '')
            medium     = str(item[2] or '')
            language   = str(item[3] or '')
            sentiment  = str(item[4] or '').strip().lower()
            subtopic   = str(item[5] or '')
            head       = str(item[6] or '')
            summary    = str(item[7] or '')

            TOPIC_MAPPING[article_id]     = broad_topic
            SENTIMENT_MAPPING[article_id] = sentiment

            ITEMS_ARTICLES.append({
                'id':           item[0],
                'headline':     head,
                'summary':      summary,
                'broad_topic':  broad_topic,
                'subtopic':     subtopic,
                'sentiment':    sentiment,
                'pubtime_month': pubtime[:7] if len(pubtime) >= 7 else '',
                'year':         int(pubtime[:4]) if len(pubtime) >= 4 and pubtime[:4].isdigit() else None,
                'date':         pubtime[:10],
                'source':       medium,
                'outlet':       SOURCE_NAME_MAP.get(medium, medium),
                'color':        SOURCE_COLOR_MAP.get(medium, '#7A7A8A'),
                'language':     language,
                'classification': broad_topic,
            })

    print(f'Loaded topic+sentiment mapping for {len(TOPIC_MAPPING):,} articles')
    print(f'ITEMS_ARTICLES: {len(ITEMS_ARTICLES):,} full records available')

load_topic_mapping()


# ── Load all data at startup ────────────────────────────────────────────────
ALL_ARTICLES = []

def parse_date(pubtime):
    if not pubtime:
        return None, ''
    try:
        import re
        dt_str = re.sub(r'([+-])(\d{2})$', r'\1\2:00', str(pubtime).strip())
        dt = datetime.fromisoformat(dt_str)
        return dt.year, dt.strftime('%d.%m.%Y')
    except Exception:
        try:
            return int(str(pubtime)[:4]), str(pubtime)[:10]
        except Exception:
            return None, ''

for lang, filename in DATA_FILES.items():
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f'WARNING: data file not found: {filepath}')
        continue
    with open(filepath, 'r', encoding='utf-8') as f:
        records = json.load(f)
    count = 0
    for item in records:
        classification = (item.get('classification') or '').strip()
        if classification.upper() in ('NOT TOPIC', 'NONE', ''):
            continue
        year, date_str = parse_date(item.get('pubtime'))
        pubtime = str(item.get('pubtime') or '')
        pubtime_month = pubtime[:7] if len(pubtime) >= 7 else ''
        article_id = str(item.get('id'))

        ALL_ARTICLES.append({
            'id':             item.get('id'),
            'headline':       item.get('head', ''),
            'summary':        item.get('summary', ''),
            'classification': classification,
            'broad_topic':    TOPIC_MAPPING.get(article_id, ''),
            'sentiment':      SENTIMENT_MAPPING.get(article_id, ''),
            'pubtime_month':  pubtime_month,
            'verification':   item.get('verification', ''),
            'source':         item.get('medium_code', ''),
            'outlet':         get_source_display(item.get('medium_code', '')),
            'color':          get_source_color(item.get('medium_code', '')),
            'language':       lang,
            'lang_label':     LANG_LABELS[lang],
            'lang_color':     LANG_COLORS[lang],
            'year':           year,
            'date':           date_str,
        })
        count += 1
    print(f'  ✓ {lang}: {count:,} topic articles loaded from {filename}')

print(f'\nTotal loaded: {len(ALL_ARTICLES):,} topic articles across {len(DATA_FILES)} languages')
print(f'Data directory: {DATA_DIR}')


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')


# ── Serve docs/data/ as /data/ for the frontend ────────────────────────────
@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(DOCS_DATA_DIR, filename)


# ── Dashboard data routes ───────────────────────────────────────────────────

@app.route('/api/dashboard/preproc')
def dashboard_preproc():
    filepath = os.path.join(DOCS_DATA_DIR, 'preproc.json')
    if not os.path.exists(filepath):
        return jsonify({'error': 'preproc.json not found'}), 404
    with open(filepath, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))

@app.route('/api/dashboard/summaries')
def dashboard_summaries():
    filepath = os.path.join(DOCS_DATA_DIR, 'summaries.json')
    if not os.path.exists(filepath):
        return jsonify({'error': 'summaries.json not found'}), 404
    with open(filepath, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))

@app.route('/api/dashboard/items_index')
def dashboard_items_index():
    filepath = os.path.join(DOCS_DATA_DIR, 'items_index.json')
    if not os.path.exists(filepath):
        return jsonify({'error': 'items_index.json not found'}), 404
    with open(filepath, 'r', encoding='utf-8') as f:
        index = json.load(f)
    rewritten = {}
    for cls, url in index.items():
        filename = os.path.basename(url)
        rewritten[cls] = f'/api/dashboard/items/{filename}'
    return jsonify(rewritten)

@app.route('/api/dashboard/items/<filename>')
def dashboard_items(filename):
    filepath = os.path.join(DOCS_DATA_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': f'{filename} not found'}), 404
    with open(filepath, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))


# ── Existing API routes ─────────────────────────────────────────────────────

@app.route('/api/articles')
def api_articles():
    q              = request.args.get('q', '').strip().lower()
    langs          = request.args.get('lang', '').strip()
    year           = request.args.get('year', type=int)
    month_from     = request.args.get('month_from', '').strip()
    month_to       = request.args.get('month_to', '').strip()
    source         = request.args.get('source', '').strip()
    classification = request.args.get('classification', '').strip().lower()
    sentiment      = request.args.get('sentiment', '').strip().lower()
    limit          = request.args.get('limit', 100, type=int)
    offset         = request.args.get('offset', 0, type=int)

    lang_list = [l.strip() for l in langs.split(',') if l.strip()] if langs else []

    pool = ITEMS_ARTICLES if (sentiment or classification) else ALL_ARTICLES

    results = []
    for article in pool:
        if lang_list and article.get('language') not in lang_list:
            continue
        if year and article.get('year') != year:
            continue
        if month_from and (article.get('pubtime_month') or '') < month_from:
            continue
        if month_to and (article.get('pubtime_month') or '') > month_to:
            continue
        if source and article.get('source') != source:
            continue
        if classification and (article.get('broad_topic') or '').lower() != classification:
            continue
        if sentiment and (article.get('sentiment') or '').lower() != sentiment:
            continue
        if q:
            searchable = ' '.join([
                article.get('headline') or '',
                article.get('summary') or '',
                article.get('broad_topic') or '',
                article.get('classification') or '',
            ]).lower()
            if q not in searchable:
                continue
        results.append(article)

    total = len(results)
    page  = results[offset:offset + limit]

    return jsonify({
        'total':    total,
        'offset':   offset,
        'limit':    limit,
        'articles': page,
    })


@app.route('/api/stats')
def api_stats():
    by_lang = {}
    by_year = {}
    for a in ALL_ARTICLES:
        l = a['language']
        by_lang[l] = by_lang.get(l, 0) + 1
        if a['year']:
            by_year[a['year']] = by_year.get(a['year'], 0) + 1

    sources = set(a['source'] for a in ALL_ARTICLES)
    years   = [a['year'] for a in ALL_ARTICLES if a['year']]
    peak_year = max(by_year, key=by_year.get) if by_year else None

    return jsonify({
        'total_articles':   len(ALL_ARTICLES),
        'total_sources':    len(sources),
        'year_min':         min(years) if years else None,
        'year_max':         max(years) if years else None,
        'by_language':      by_lang,
        'by_year':          by_year,
        'peak_year':        peak_year,
        'peak_year_count':  by_year.get(peak_year, 0) if peak_year else 0,
    })


@app.route('/api/classifications')
def api_classifications():
    lang = request.args.get('lang', '').strip()
    counts = {}
    for a in ALL_ARTICLES:
        if lang and a['language'] != lang:
            continue
        c = a['classification']
        if c and c.upper() not in ('NOT TOPIC', 'NONE', ''):
            counts[c] = counts.get(c, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    return jsonify([{'label': k, 'count': v} for k, v in sorted_counts])


@app.route('/api/sources')
def api_sources():
    lang = request.args.get('lang', '').strip()
    counts = {}
    for a in ALL_ARTICLES:
        if lang and a['language'] != lang:
            continue
        s = a['source']
        if s:
            counts[s] = counts.get(s, 0) + 1
    result = []
    for code, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count >= 2:
            result.append({
                'id':    code,
                'name':  get_source_display(code),
                'color': get_source_color(code),
                'count': count,
            })
    return jsonify(result)


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('frontend', filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)