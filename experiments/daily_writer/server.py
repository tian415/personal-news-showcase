"""Loopback-only reader and version-bound feedback endpoint; no provider calls."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .feedback_store import FeedbackStore
from .pipeline import digest


def handler(root, port, page_name):
    request = json.loads((root / 'request.json').read_text())
    response = json.loads(json.loads((root / 'model-response.json').read_text())['content'])
    if (root / 'reader-response.json').exists():
        response = json.loads((root / 'reader-response.json').read_text())
    if response['request_hash'] != request['request_hash']:
        raise ValueError('draft binding mismatch')
    store = FeedbackStore(root, request['request_hash'], digest(response),
                          [i['id'] for i in request['packet']['items']])
    host = f'127.0.0.1:{port}'

    class Reader(BaseHTTPRequestHandler):
        timeout = 5

        def send_bytes(self, data, kind='application/json; charset=utf-8'):
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.headers.get('Host') != host:
                self.send_error(403)
                return
            path = urlsplit(self.path).path
            if path in ('/', '/' + page_name):
                self.send_bytes((root / page_name).read_bytes(), 'text/html; charset=utf-8')
            elif path == '/feedback':
                try:
                    self.send_bytes(json.dumps(store.load()).encode())
                except (ValueError, OSError):
                    self.send_error(409, 'Feedback unavailable; existing data preserved')
            else:
                self.send_error(404)

        def do_POST(self):
            if (self.headers.get('Host') != host or
                    self.headers.get('Origin') != 'http://' + host or
                    self.path != '/feedback' or
                    self.headers.get('Content-Type') != 'application/json'):
                self.send_error(403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 32768:
                    raise ValueError('invalid size')
                value = json.loads(self.rfile.read(size))
                store.save(value)
                self.send_bytes(json.dumps({'saved': True, 'draft_hash': store.draft_hash}).encode())
            except (ValueError, TypeError, KeyError, OSError):
                self.send_error(400, 'Invalid feedback; not saved')

    return Reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8774)
    parser.add_argument('--page', choices=['preview.html', 'rejected-preview.html'],
                        default='rejected-preview.html')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port),
                                handler(args.root.resolve(), args.port, args.page))
    server.daemon_threads = True
    server.serve_forever()


if __name__ == '__main__':
    main()
