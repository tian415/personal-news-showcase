"""Manual local entrypoint; prepare and ingest never call a provider."""
import argparse
import json
from pathlib import Path

from .pipeline import frozen_packet, generate, ingest, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'ingest', 'generate', 'feedback-context'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--response', type=Path)
    parser.add_argument('--authorize-request', help='Use only after explicit human paid-call approval')
    parser.add_argument('--feedback-from', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.command == 'prepare':
        packet = frozen_packet(Path(__file__).resolve().parents[2])
        if args.feedback_from:
            from .feedback_store import review_context
            packet['reader_feedback_context'] = review_context(args.feedback_from)
        result = prepare(args.root, packet)
        print(json.dumps({'request_hash': result['request_hash'], 'status': 'no_model_call'}))
    elif args.command == 'feedback-context':
        from .feedback_store import review_context
        from .pipeline import private_root, write
        write(private_root(args.root) / 'reader-context.json', review_context(args.feedback_from))
        print('Local reader context saved; no model call or ranking change')
    elif args.command == 'ingest':
        if args.response is None:
            parser.error('--response required')
        print(json.dumps(ingest(args.root, json.loads(args.response.read_text()))))
    else:
        if not args.authorize_request:
            parser.error('explicit --authorize-request required after paid-call approval')
        from news_assistant.credential_store import load_deepseek_key
        print(json.dumps(generate(args.root, authorized_hash=args.authorize_request,
                                  api_key=load_deepseek_key())))


if __name__ == '__main__':
    main()
