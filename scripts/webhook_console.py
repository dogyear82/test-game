#!/usr/bin/env python3
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from email.parser import BytesParser
from email.policy import default as email_default


class Handler(BaseHTTPRequestHandler):
    max_bytes = 200_000  # limit console spam
    verbose = False

    def _log(self, msg):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] {msg}")

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0') or 0)
        ctype = self.headers.get('Content-Type', '')
        body = self.rfile.read(length) if length > 0 else b''

        self._log(f"POST {self.path} from {self.client_address[0]} ct={ctype} len={length}")
        if self.verbose:
            self._log(f"Headers: {dict(self.headers)}")

        printed = False
        try:
            if 'application/json' in ctype:
                data = json.loads(body.decode('utf-8', errors='ignore'))
                print(json.dumps(data, indent=2))
                printed = True
            elif 'multipart/form-data' in ctype:
                # Build a minimal MIME message for parsing
                raw = ("Content-Type: ".encode('utf-8') + ctype.encode('utf-8') +
                       b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
                msg = BytesParser(policy=email_default).parsebytes(raw)
                if msg.is_multipart():
                    for part in msg.iter_parts():
                        self._print_multipart_part(part)
                    printed = True
        except Exception as e:
            self._log(f"parse error: {e}")

        if not printed:
            # raw body (truncated)
            print(body[: self.max_bytes])
            if len(body) > self.max_bytes:
                self._log(f"(truncated {len(body) - self.max_bytes} bytes)")

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def _print_multipart_part(self, part):
        cd = part.get('Content-Disposition', '') or ''
        name = part.get_param('name', header='Content-Disposition')
        filename = part.get_filename()
        ctype = part.get_content_type()
        payload = part.get_payload(decode=True) or b''
        if filename:
            self._log(f"file field '{name or ''}': name={filename!r}, size={len(payload)} bytes, type={ctype}")
        else:
            text = payload.decode('utf-8', errors='ignore')
            if (name or '').lower() == 'payload':
                try:
                    obj = json.loads(text)
                    self._log(f"field '{name}': JSON ->\n" + json.dumps(obj, indent=2))
                    return
                except Exception:
                    pass
            self._log(f"field '{name or ''}': {text[:512]}" + (" ..." if len(text) > 512 else ""))

    def log_message(self, fmt, *args):
        # silence default access logs
        if self.verbose:
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    Handler.verbose = args.verbose
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Webhook console server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == '__main__':
    main()
