from flask import Flask, send_from_directory, jsonify, request
import pandas as pd
import os

app = Flask(__name__, static_folder='frontend')

# ── Load data once at startup ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'swiss_ai_discourse_articles_de.csv')

df = pd.read_csv(DATA_PATH)
df['pubdate'] = pd.to_datetime(df['pubdate'], errors='coerce')
df = df.dropna(subset=['pubdate'])
df['year'] = df['pubdate'].dt.year

# Source display name mapping (Swissdox codes → readable names)
SOURCE_NAMES = {
    'NZZ': 'NZZ', 'NZZM': 'NZZ am Sonntag', 'NZZO': 'NZZ Online',
    'TA': 'Tages-Anzeiger', 'TAM': 'Tages-Anzeiger', 'TASI': 'Tages-Anzeiger',
    'SRF': 'SRF', 'SRFA': 'SRF', 'SRFV': 'SRF',
    'BLI': 'Blick', 'BLIO': 'Blick Online',
    'AZ': 'Aargauer Zeitung', 'AZO': 'Aargauer Zeitung Online',
    'BZ': 'Basler Zeitung', 'BAZ': 'Basler Zeitung',
    'BEOL': 'Berner Zeitung', 'BEOLO': 'Berner Zeitung Online',
    'WOZ': 'WOZ', 'SF': 'Sonntagszeitung', 'SONT': 'Sonntagszeitung',
    'CASH': 'Cash', 'CASO': 'Cash Online',
    'FUW': 'Finanz und Wirtschaft', 'FUWO': 'Finanz und Wirtschaft Online',
    'SGT': 'St. Galler Tagblatt', 'LUZ': 'Luzerner Zeitung',
    'SHZ': 'Schweizer Handelszeitung',
}

SOURCE_TYPES = {
    'NZZ': 'National', 'NZZM': 'National', 'NZZO': 'National',
    'TA': 'National', 'TAM': 'National',
    'SRF': 'Öffentl. Rundfunk', 'SRFA': 'Öffentl. Rundfunk', 'SRFV': 'Öffentl. Rundfunk',
    'BLI': 'Boulevard', 'BLIO': 'Boulevard',
    'AZ': 'Regional', 'AZO': 'Regional',
    'BZ': 'Regional', 'BAZ': 'Regional',
    'BEOL': 'Regional', 'BEOLO': 'Regional',
    'SGT': 'Regional', 'LUZ': 'Regional', 'SHZ': 'Regional',
    'WOZ': 'Unabhängig', 'FUW': 'Wirtschaft', 'FUWO': 'Wirtschaft',
    'CASH': 'Wirtschaft', 'CASO': 'Wirtschaft',
}

SOURCE_COLORS = {
    'NZZ': '#C0B8E8', 'NZZM': '#C0B8E8', 'NZZO': '#C0B8E8',
    'TA': '#7BA4D4',  'TAM': '#7BA4D4',
    'SRF': '#E87070', 'SRFA': '#E87070', 'SRFV': '#E87070',
    'BLI': '#D45B5B', 'BLIO': '#D45B5B',
    'AZ': '#7BB4D4',  'AZO': '#7BB4D4',
    'BZ': '#8BBCAA',  'BAZ': '#8BBCAA',
    'WOZ': '#C4A882', 'SGT': '#A882C4',
    'FUW': '#E8C060', 'FUWO': '#E8C060',
}

def get_source_display(code):
    return SOURCE_NAMES.get(code, code)

def get_source_type(code):
    return SOURCE_TYPES.get(code, 'Andere')

def get_source_color(code):
    return SOURCE_COLORS.get(code, '#7A7A8A')


# ── Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')


@app.route('/api/sources')
def api_sources():
    """Return all sources with article counts, sorted by count."""
    counts = df['source'].value_counts().reset_index()
    counts.columns = ['source', 'count']
    # Only include sources with >5 articles
    counts = counts[counts['count'] > 5]
    result = []
    for _, row in counts.iterrows():
        code = row['source']
        result.append({
            'id':    code,
            'name':  get_source_display(code),
            'type':  get_source_type(code),
            'color': get_source_color(code),
            'count': int(row['count']),
        })
    return jsonify(result)


@app.route('/api/articles')
def api_articles():
    """
    Return articles filtered by query, sources, year range.
    Query params:
        q        – search term (searches headline + subtitle)
        sources  – comma-separated source codes, e.g. NZZ,TA,SRF
        year_from – e.g. 2022
        year_to   – e.g. 2024
        limit    – max results (default 50)
    """
    q         = request.args.get('q', '').strip()
    sources   = request.args.get('sources', '').strip()
    year_from = request.args.get('year_from', type=int)
    year_to   = request.args.get('year_to',   type=int)
    limit     = request.args.get('limit', 50, type=int)

    filtered = df.copy()

    if q:
        mask = (
            filtered['head'].str.contains(q, case=False, na=False) |
            filtered['title_full'].str.contains(q, case=False, na=False) |
            filtered['subtitle'].str.contains(q, case=False, na=False)
        )
        filtered = filtered[mask]

    if sources:
        src_list = [s.strip() for s in sources.split(',') if s.strip()]
        filtered = filtered[filtered['source'].isin(src_list)]

    if year_from:
        filtered = filtered[filtered['year'] >= year_from]
    if year_to:
        filtered = filtered[filtered['year'] <= year_to]

    filtered = filtered.sort_values('pubdate', ascending=False).head(limit)

    result = []
    for _, row in filtered.iterrows():
        result.append({
            'headline':  row['head']       or row['title_full'] or '',
            'subtitle':  row['subtitle']   or '',
            'date':      row['pubdate'].strftime('%d.%m.%Y') if pd.notna(row['pubdate']) else '',
            'source':    row['source'],
            'outlet':    get_source_display(row['source']),
            'color':     get_source_color(row['source']),
            'url':       row['articleURL'] if pd.notna(row['articleURL']) else None,
        })
    return jsonify(result)


@app.route('/api/sources_for_query')
def api_sources_for_query():
    """
    Return sources that have articles matching a search query,
    with per-source article counts.
    Query params: q (required)
    """
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify([])

    mask = (
        df['head'].str.contains(q, case=False, na=False) |
        df['title_full'].str.contains(q, case=False, na=False) |
        df['subtitle'].str.contains(q, case=False, na=False)
    )
    matched = df[mask]

    counts = matched['source'].value_counts().reset_index()
    counts.columns = ['source', 'count']

    result = []
    for _, row in counts.iterrows():
        code = row['source']
        result.append({
            'id':    code,
            'name':  get_source_display(code),
            'type':  get_source_type(code),
            'color': get_source_color(code),
            'count': int(row['count']),
        })
    return jsonify(result)


@app.route('/api/stats')
def api_stats():
    """Return high-level dataset statistics for the UI header."""
    return jsonify({
        'total_articles': len(df),
        'total_sources':  int(df['source'].nunique()),
        'year_min':       int(df['year'].min()),
        'year_max':       int(df['year'].max()),
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)