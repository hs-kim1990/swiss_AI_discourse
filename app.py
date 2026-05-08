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


# ── Load all data at startup ────────────────────────────────────────────────
ALL_ARTICLES = []

def parse_date(pubtime):
    if not pubtime:
        return None, ''
    try:
        dt_str = str(pubtime).strip()
        # Normalise timezone: +01 → +01:00, handle both + and - offsets
        import re
        dt_str = re.sub(r'([+-])(\d{2})$', r'\1\2:00', dt_str)
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
        ALL_ARTICLES.append({
            'id':             item.get('id'),
            'headline':       item.get('head', ''),
            'summary':        item.get('summary', ''),
            'classification': classification,
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
if len(ALL_ARTICLES) == 0:
    print('ERROR: No articles loaded — check that data files exist in the data/ folder')


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')


# ── Dashboard data routes ───────────────────────────────────────────────────

@app.route('/api/dashboard/preproc')
def dashboard_preproc():
    """Serve preproc.json from docs/data/"""
    filepath = os.path.join(DOCS_DATA_DIR, 'preproc.json')
    if not os.path.exists(filepath):
        return jsonify({'error': 'preproc.json not found'}), 404
    with open(filepath, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))

@app.route('/api/dashboard/summaries')
def dashboard_summaries():
    """Serve summaries.json from docs/data/"""
    filepath = os.path.join(DOCS_DATA_DIR, 'summaries.json')
    if not os.path.exists(filepath):
        return jsonify({'error': 'summaries.json not found'}), 404
    with open(filepath, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))

@app.route('/api/dashboard/items_index')
def dashboard_items_index():
    """Serve items_index.json from docs/data/, rewriting item URLs to use Flask routes"""
    filepath = os.path.join(DOCS_DATA_DIR, 'items_index.json')
    if not os.path.exists(filepath):
        return jsonify({'error': 'items_index.json not found'}), 404
    with open(filepath, 'r', encoding='utf-8') as f:
        index = json.load(f)
    # Rewrite each value to go through Flask so relative paths work
    rewritten = {}
    for cls, url in index.items():
        # Extract just the filename from whatever path is stored
        filename = os.path.basename(url)
        rewritten[cls] = f'/api/dashboard/items/{filename}'
    return jsonify(rewritten)

@app.route('/api/dashboard/items/<filename>')
def dashboard_items(filename):
    """Serve individual topic items JSON files from docs/data/"""
    filepath = os.path.join(DOCS_DATA_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': f'{filename} not found'}), 404
    with open(filepath, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))


# ── Existing API routes ─────────────────────────────────────────────────────

@app.route('/api/articles')
def api_articles():
    q      = request.args.get('q', '').strip().lower()
    langs  = request.args.get('lang', '').strip()
    year   = request.args.get('year', type=int)
    source = request.args.get('source', '').strip()
    limit  = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)

    lang_list = [l.strip() for l in langs.split(',') if l.strip()] if langs else []

    results = []
    for article in ALL_ARTICLES:
        if lang_list and article['language'] not in lang_list:
            continue
        if year and article['year'] != year:
            continue
        if source and article['source'] != source:
            continue
        if q:
            searchable = ' '.join([
                article.get('headline') or '',
                article.get('summary') or '',
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
    for a in ALL_ARTICLES:
        l = a['language']
        by_lang[l] = by_lang.get(l, 0) + 1

    sources = set(a['source'] for a in ALL_ARTICLES)
    years   = [a['year'] for a in ALL_ARTICLES if a['year']]

    return jsonify({
        'total_articles': len(ALL_ARTICLES),
        'total_sources':  len(sources),
        'year_min':       min(years) if years else None,
        'year_max':       max(years) if years else None,
        'by_language':    by_lang,
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