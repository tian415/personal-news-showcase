"""Explicit checkout command for the private local chain."""
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from . import _failed_edition, _root, approve, collect, import_feedback, prepare, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('collect', 'prepare', 'run', 'approve', 'import-feedback'):
        child = sub.add_parser(command)
        child.add_argument('--root', type=Path, required=True, help='Private run directory')
        if command == 'collect':
            child.add_argument('--fetch', action='store_true', help='Explicit public network collection')
            child.add_argument('--replay', action='store_true')
            child.add_argument('--reference-utc', default=None)
            child.add_argument('--sources', type=Path)
        if command in {'prepare', 'run'}:
            child.add_argument('--corpus', type=Path)
        if command == 'prepare':
            child.add_argument('--feedback-from', type=Path, action='append', default=[])
        if command == 'run':
            for name in ('reviews', 'facts', 'packets', 'supplements'):
                child.add_argument('--' + name, type=Path)
            child.add_argument('--limit', type=int, default=12)
        if command == 'approve':
            child.add_argument('--reviewer', required=True)
            child.add_argument('--expected-hash', required=True)
        if command == 'import-feedback':
            child.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'collect':
            _failed_edition(_root(args.root))
            if not args.fetch and not args.replay:
                parser.error('collection requires --fetch or --replay; offline default')
            if args.replay and args.reference_utc is None:
                args.reference_utc = json.loads((args.root / 'corpus/corpus.json').read_text())['reference_utc']
            reference = datetime.fromisoformat(args.reference_utc) if args.reference_utc else datetime.now(UTC)
            sources = json.loads(args.sources.read_text()) if args.sources else None
            result = collect(args.root, reference, sources=sources, replay=args.replay)
        elif args.command == 'prepare':
            result = prepare(args.root, args.corpus or args.root / 'corpus/corpus.json',
                             feedback_roots=args.feedback_from)
        elif args.command == 'run':
            result = run(args.root, corpus=args.corpus or args.root / 'corpus/corpus.json',
                         reviews=args.reviews or args.root / 'review/reviews.json',
                         facts=args.facts or args.root / 'review/facts.json',
                         packets=args.packets or args.root / 'review/packets.json',
                         supplements=args.supplements, limit=args.limit)
            result = {'status': result['status'], 'edition_hash': result['edition_hash'],
                      'selected_events': len(result['result']['selected']),
                      'draft': str(args.root / 'review-preview.html')}
        elif args.command == 'approve':
            result = approve(args.root, reviewer=args.reviewer, expected_hash=args.expected_hash)
        else:
            result = import_feedback(args.root, args.input)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(1, f'Blocked: {error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
