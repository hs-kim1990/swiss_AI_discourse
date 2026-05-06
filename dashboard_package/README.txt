Swiss AI Discourse Dashboard — 10 Million Initiative
=====================================================

QUICK START
-----------
Windows : Double-click run.bat
Mac/Linux: ./run.sh   (or: python3 run.py)

The browser opens automatically at http://localhost:8765

REQUIREMENTS
------------
Python 3.7 or later (standard library only — no pip installs needed).
A modern browser (Chrome, Firefox, Edge, Safari).

CONTENTS
--------
index.html          Main dashboard
run.py              Server launcher (cross-platform)
run.bat             Windows launcher
run.sh              Mac/Linux launcher
data/
  preproc.json      Timeline + topic statistics (208 KB)
  items.json        Article index, 23,785 items (9.7 MB)
  summaries.json    AI-generated argument summaries (99 KB)

DATA NOTES
----------
* All "NOT TOPIC" and null-verification items have been removed.
* Valid items: 23,785 out of 124,191 raw records.
* Date range: Oct 2023 – Mar 2026.
* Languages: German (de), French (fr), Italian (it).
* All content is framed around the "Pas de Suisse à 10 millions!"
  initiative — the UDC-led movement to limit Switzerland's population
  to 10 million through immigration restriction.

STOPPING THE SERVER
-------------------
Press Ctrl+C in the terminal window, or close the terminal.
