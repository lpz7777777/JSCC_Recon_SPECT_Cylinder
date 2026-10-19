"""Read-only project/Ac225 directory catalog; never imports experiment runners."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT/'docs/ac225/catalog.json'
SOURCES = ROOT/'docs/ac225/source_index.json'
DIRECTORIES = ROOT/'docs/ac225/directory_index.json'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def matches(item, query):
    return query.casefold() in json.dumps(item, ensure_ascii=False).casefold()


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Catalog paths must remain project-relative: '+value)
    return ROOT/path


def check(catalog, sources, directories=None):
    errors = []
    optional_missing = []
    ids = [item['id'] for item in catalog['experiments']]
    if len(ids) != len(set(ids)):
        errors.append('Duplicate experiment ids')
    for item in catalog['experiments']+catalog['components']:
        for key in ['entries', 'evidence', 'local_data', 'figures']:
            for value in item.get(key, []):
                try:
                    path = relative_path(value)
                except ValueError as error:
                    errors.append(str(error))
                    continue
                if not path.exists():
                    (optional_missing if key in ['local_data', 'figures'] else errors).append(value)
    for item in sources['files']:
        if not relative_path(item['path']).is_file():
            errors.append(item['path'])
    for item in (directories or {}).get('directories', []):
        if not relative_path(item['path']).is_dir():
            (optional_missing if item['local_only'] else errors).append(item['path'])
    return dict(passed=not errors, experiments=len(ids), indexed_sources=len(sources['files']),
                indexed_directories=len((directories or {}).get('directories', [])),
                errors=errors, local_only_missing=optional_missing,
                status_semantics='Dated local evidence snapshot; no SSH query or job action')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['list', 'show', 'find', 'files', 'dirs', 'check']:
        command = sub.add_parser(name)
        command.add_argument('--json', action='store_true')
        if name in ['list', 'files', 'dirs']:
            command.add_argument('--topic', default='')
        if name == 'show':
            command.add_argument('id')
        if name == 'find':
            command.add_argument('query')
        if name == 'files':
            command.add_argument('--group', default='')
    args = parser.parse_args()
    catalog, sources, directories = load(CATALOG), load(SOURCES), load(DIRECTORIES)
    if args.command == 'check':
        result = check(catalog, sources, directories)
    elif args.command == 'show':
        found = [item for item in catalog['experiments']+catalog['components'] if item['id'] == args.id]
        if not found:
            parser.error('Unknown id; use list or find')
        result = found[0]
    elif args.command == 'list':
        result = [item for item in catalog['experiments'] if matches(item, args.topic)]
    elif args.command == 'files':
        result = [item for item in sources['files']
                  if (not args.group or item['group'] == args.group) and matches(item, args.topic)]
    elif args.command == 'dirs':
        result = [item for item in directories['directories'] if matches(item, args.topic)]
    else:
        result = dict(experiments=[item for item in catalog['experiments']+catalog['components'] if matches(item, args.query)],
                      files=[item for item in sources['files'] if matches(item, args.query)],
                      directories=[item for item in directories['directories'] if matches(item, args.query)])
    if args.json or args.command in ['show', 'check', 'find']:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for item in result:
            if args.command == 'list':
                print(f"{item['id']:<33} {item['status']:<22} {item['title']}")
            elif args.command == 'dirs':
                print(f"{item['role']:<35} {item['path']}")
            else:
                print(f"{item['group']:<22} {item['path']}  {item['purpose']}")
    if args.command == 'check' and not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
