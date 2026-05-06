#!/usr/bin/env python3
"""
Swiss AI Discourse Dashboard — local server launcher.
Run:  python run.py
Then open http://localhost:8765 in your browser (it opens automatically).
"""
import http.server, socketserver, webbrowser, os, sys, threading, time

PORT = 8765
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

def open_browser():
    time.sleep(0.8)
    webbrowser.open(f"http://localhost:{PORT}/index.html")

print("=" * 55)
print("  Swiss AI Discourse — 10 Million Initiative")
print("  Dashboard Server")
print("=" * 55)
print(f"  Starting server on http://localhost:{PORT}")
print("  Opening browser automatically...")
print("  Press Ctrl+C to stop.\n")

threading.Thread(target=open_browser, daemon=True).start()

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\n  Server stopped.")
    sys.exit(0)
except OSError as e:
    print(f"\n  Port {PORT} is already in use. Try: python run.py --port 8766")
    sys.exit(1)
