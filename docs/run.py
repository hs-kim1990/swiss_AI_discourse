#!/usr/bin/env python3
"""
Swiss AI Discourse Dashboard — local server launcher.
Run:  python run.py
Then open http://localhost:8765 in your browser (it opens automatically).
"""
import http.server
import socketserver
import webbrowser
import os
import sys
import threading
import time
import argparse

PORT = 8765
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

def open_browser():
    time.sleep(0.8)
    webbrowser.open(f"http://localhost:{PORT}/index.html")

parser = argparse.ArgumentParser(description='Launch local dashboard server')
parser.add_argument('-p', '--port', type=int, default=PORT, help='Port to listen on (default: 8765)')
parser.add_argument('--try-range', type=int, default=10, help='Number of consecutive ports to try if the first is busy')
args = parser.parse_args()
start_port = args.port
max_tries = max(1, args.try_range)

for p in range(start_port, start_port + max_tries):
    try:
        PORT = p
        print("=" * 55)
        print("  Swiss AI Discourse — 10 Million Initiative")
        print("  Dashboard Server")
        print("=" * 55)
        print(f"  Starting server on http://localhost:{PORT}")
        print("  Opening browser automatically...")
        print("  Press Ctrl+C to stop.\n")

        threading.Thread(target=open_browser, daemon=True).start()

        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        sys.exit(0)
    except OSError:
        print(f"\n  Port {p} is already in use, trying next port...")
        continue

print(f"\n  No available ports found in range {start_port}-{start_port+max_tries-1}.")
sys.exit(1)
