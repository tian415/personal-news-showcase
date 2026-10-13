"""One HTML body per supervised process; never share native parser state across threads.

A nonzero native exit aborts the batch, rather than producing an empty article or
retrying unknown work. This isolates native faults; it does not patch libxml2.
"""
import json
import subprocess
import sys
from pathlib import Path


class ExtractionFailure(RuntimeError):
    """The isolated parser failed; the enclosing edition must stop."""


def extract_body(raw: bytes) -> str:
    command = [sys.executable, '-X', 'faulthandler', str(Path(__file__).resolve())]
    try:
        result = subprocess.run(command, input=raw, capture_output=True, timeout=60, check=False)
    except subprocess.TimeoutExpired as error:
        raise ExtractionFailure('native extraction timeout (60s); no retry') from error
    except OSError as error:
        raise ExtractionFailure(f'native worker could not start: {error}') from error
    if result.returncode:
        detail = result.stderr.decode('utf-8', errors='replace')[-8192:]
        raise ExtractionFailure(f'native extraction exit {result.returncode}; no retry\n{detail}')
    try:
        value = json.loads(result.stdout)
    except (ValueError, UnicodeError) as error:
        raise ExtractionFailure('invalid native worker output') from error
    if not isinstance(value, str):
        raise ExtractionFailure('invalid native worker output type')
    return value


def main():
    # Imported only in the child. The parent sends bytes and receives text, never lxml trees.
    import trafilatura
    raw = sys.stdin.buffer.read()
    value = trafilatura.extract(raw, include_comments=False, include_tables=True) or ''
    sys.stdout.buffer.write(json.dumps(value, ensure_ascii=False).encode('utf-8'))


if __name__ == '__main__':
    main()
