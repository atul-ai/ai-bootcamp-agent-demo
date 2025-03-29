#!/usr/bin/env python3
"""
Simple HTTP server for the arXiv Assistant frontend.
"""
import http.server
import socketserver
import os

# Configuration
PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler with CORS support."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        """Add CORS headers."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests."""
        self.send_response(200)
        self.end_headers()

if __name__ == "__main__":
    print(f"Starting frontend server on http://localhost:{PORT}")
    print(f"Press Ctrl+C to stop the server.")
    
    # Create the server
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            # Serve until interrupted
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down the server...")
            httpd.server_close() 