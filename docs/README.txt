Swiss AI Discourse Dashboard — 10 Million Initiative
=====================================================

────────────────────────────────────────────────────
RUN LOCALLY
────────────────────────────────────────────────────
Windows : Double-click  run.bat
Mac/Linux: ./run.sh   (or: python3 run.py)

Opens automatically at http://localhost:8765
Requires Python 3.7+ (no external packages needed).


────────────────────────────────────────────────────
PUBLISH ON GITHUB PAGES (free, public URL)
────────────────────────────────────────────────────

Step 1 — Create a GitHub account if you don't have one
  https://github.com/signup

Step 2 — Create a new repository
  https://github.com/new
  Name suggestion: swiss-ai-discourse
  Visibility: Public  (required for free GitHub Pages)
  Do NOT initialise with README (you already have files)

Step 3 — Push this folder to the repo
  Open a terminal inside this folder, then run:

    git init
    git add .
    git commit -m "Initial dashboard"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/swiss-ai-discourse.git
    git push -u origin main

  Replace YOUR_USERNAME with your GitHub username.

Step 4 — Enable GitHub Pages
  Go to your repo → Settings → Pages
  Source: "Deploy from a branch"
  Branch: main  /  (root)
  Click Save

Step 5 — Your site is live in ~60 seconds at:
  https://YOUR_USERNAME.github.io/swiss-ai-discourse/

No server needed — GitHub Pages serves the static files directly.
The dashboard fetches its data files automatically from the same URL.


────────────────────────────────────────────────────
DATA SIZE & LOAD STRATEGY
────────────────────────────────────────────────────
                                             Size
  index.html                                37 KB  <- loads instantly
  data/preproc.json   (stats + timeline)   208 KB  <- loads on startup
  data/summaries.json (AI bullet points)    99 KB  <- loads on startup
  data/items_index.json                      1 KB  <- loads on startup
  ─────────────────────────────────────────────────
  Initial page load                        ~345 KB <- very fast

Topic article files load ON DEMAND when you open a
topic in the Detail view (cached after first load):
  items_economy_labor_finance.json          2.4 MB
  items_swiss_domestic_politics_...json     1.7 MB
  items_healthcare_social_welfare_...json   1.2 MB
  items_environment_energy_...json          1.0 MB
  items_immigration_asylum_policy.json      0.9 MB
  items_demographic_change_...json          0.7 MB
  items_defense_security_military.json      0.6 MB
  items_swiss_eu_relations_...json          0.5 MB
  items_refugee_migrant_...json             0.4 MB
  items_international_relations_...json     0.1 MB
  items_others.json                        <0.1 MB

All files are under GitHub's 50 MB recommended limit.
Total repo size: ~10 MB (GitHub limit: 1 GB).

Source data: 124,191 raw records.
Removed: 100,406 "NOT TOPIC" / null-verification items.
Retained: 23,785 valid articles.


────────────────────────────────────────────────────
UPDATING AFTER A PUSH
────────────────────────────────────────────────────
  git add .
  git commit -m "Update summaries"
  git push

GitHub Pages redeploys automatically within ~60 seconds.
