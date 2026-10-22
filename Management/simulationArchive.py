#!/usr/bin/env python3
"""Standalone, standard-library inventory and verified local simulation archiver.

Sources are never moved or deleted. Large payloads are compressed per simulation;
configurations, CSVs, videos and other files remain directly accessible.
"""

import argparse
import csv
from datetime import datetime, timezone
import fnmatch
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tarfile
import uuid

VERSION = 1
PAYLOADS = {'data', 'dumps'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return stream_digest(stream)


def stream_digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def state(path):
    s = path.stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def frame_directory(name):
    name = name.lower()
    return name == 'frames' or name.startswith('frames_') or name.endswith('_frames')


def frame_png(path):
    return path.suffix.lower() == '.png' and any(frame_directory(p) for p in path.parts[:-1])


def files_under(source):
    """Record exact logical sizes; do not follow links or estimate from samples."""
    files, directories, excluded = {}, [], 0
    if source.is_symlink():
        raise ValueError(f'Symlinks require explicit resolution: {source}')
    if source.is_file():
        return {source.name: state(source)}, [], 0
    for folder, dirs, names in os.walk(source):
        for name in dirs + names:
            p = Path(folder) / name
            relative = p.relative_to(source)
            if p.is_symlink():
                raise ValueError(f'Symlinks require explicit resolution: {p}')
            if p.is_dir():
                directories.append(str(relative))
            elif not p.is_file():
                raise ValueError(f'Unsupported file type: {p}')
            elif frame_png(Path(source.name) / relative):
                excluded += p.stat().st_size
            else:
                files[str(relative)] = state(p)
    return files, sorted(directories), excluded


def discover(root):
    """Partition runs, nested campaigns, loose archives and other assets."""
    for p in sorted(root.iterdir()):
        if p.is_symlink():
            yield p, 'unresolved_symlink'
        elif p.is_file():
            yield p, 'asset'
        elif (p / 'macroData.csv').exists() or (p / 'config.conf').exists():
            yield p, 'simulation'
        elif p.name in PAYLOADS | {'backups'} or frame_directory(p.name):
            yield p, 'unclassified_directory'
        elif not any(p.iterdir()):
            yield p, 'empty_directory'
        else:
            yield from discover(p)


def read_config(path):
    values, version = {}, ''
    for line in path.read_text().splitlines():
        if line.startswith('# Version:'):
            version = line.partition(':')[2].strip()
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if '=' not in line:
            raise ValueError(f'Malformed config line: {line!r}')
        key, value = (part.strip() for part in line.split('=', 1))
        if not key or key in values:
            raise ValueError(f'Empty or repeated config key: {key!r}')
        values[key] = value
    return values, version


def macro_summary(path):
    """Honor repeated/#HEADER headers and explicitly flag legacy missing indices."""
    before = state(path)
    result = {'macro_rows': 0, 'legacy_missing_index_rows': 0, 'malformed_rows': 0,
              'nonmonotonic_loads': 0, 'unterminated_macro_line': False, 'macro_columns': []}
    columns, observed, loads, previous = None, set(), [], None
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for binary in stream:
            h.update(binary)
            result['unterminated_macro_line'] = not binary.endswith(b'\n')
            line = binary.decode('utf-8').strip()
            if not line:
                continue
            if columns is None or line.startswith('#HEADER:'):
                columns = next(csv.reader([line.removeprefix('#HEADER:').strip()]))
                if len(set(columns)) != len(columns):
                    raise ValueError('Repeated column names in macro header')
                observed.update(columns)
                continue
            if line.startswith('#'):
                continue
            values = next(csv.reader([line]))
            if values == columns:
                continue
            headers = columns
            if len(values) == len(columns) - 1 and columns[0] == 'load_step':
                headers = columns[1:]
                result['legacy_missing_index_rows'] += 1
            if len(values) != len(headers):
                result['malformed_rows'] += 1
                continue
            row = dict(zip(headers, values))
            key = next((k for k in ['load', 'Load'] if k in row), None)
            if key is None:
                raise ValueError('Macro header has no load column')
            load = float(row[key])
            if not math.isfinite(load):
                raise ValueError('Non-finite macro load')
            if previous is not None and load < previous:
                result['nonmonotonic_loads'] += 1
            previous = load
            if not loads:
                result['first_load'] = load
            loads[:] = [load]
            result['maximum_load'] = max(result.get('maximum_load', load), load)
            result['macro_rows'] += 1
    result['macro_columns'] = sorted(observed)
    result['last_load'] = loads[-1] if loads else None
    result['macro_sha256'] = h.hexdigest()
    result['macro_changed_during_scan'] = before != state(path)
    return result


def database(catalogue):
    catalogue.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(catalogue / 'catalogue.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS units (source_key TEXT PRIMARY KEY, archive_id TEXT UNIQUE, source_root TEXT, metadata TEXT)')
    return db


def inventory(source, catalogue, label, annotations=None, active=()):
    source, catalogue = source.resolve(), catalogue.resolve()
    if not source.is_dir() or source == catalogue or source in catalogue.parents:
        raise ValueError('Catalogue must be outside an existing source root')
    annotations = annotations or {}
    units = list(discover(source))
    relative_names = [str(p.relative_to(source)) for p, _ in units]
    if set(annotations) - set(relative_names):
        raise ValueError(f'Annotations refer to unknown sources: {set(annotations)-set(relative_names)}')
    for pattern in active:
        if not any(fnmatch.fnmatchcase(name, pattern) for name in relative_names):
            raise ValueError(f'Active-run pattern matched no source: {pattern}')
    db = database(catalogue)
    seen = []
    for index, (path, kind) in enumerate(units, 1):
        relative = str(path.relative_to(source))
        key = hashlib.sha256(f'{label}\0{source}\0{relative}'.encode()).hexdigest()
        old = db.execute('SELECT archive_id FROM units WHERE source_key=?', (key,)).fetchone()
        archive_id = old[0] if old else str(uuid.uuid4())
        row = dict(source_key=key, archive_id=archive_id, source_label=label,
                   source_root=str(source), source_relative_path=relative, kind=kind,
                   observed_at=stamp(), year='unknown', campaign='unassigned', comments='',
                   active=any(fnmatch.fnmatchcase(relative, p) for p in active),
                   config_version_evidence='', executable_version='unknown',
                   stress_convention='unknown', energy_normalization='unknown',
                   delta_E_S_tangent='unknown', scientific_quality='not_audited',
                   source_stability='not_an_atomic_snapshot', errors=[])
        config = {}
        try:
            listing, directories, excluded = files_under(path)
            # Preserve the inventory even if historical scientific metadata fails.
            row.update(file_count=len(listing), logical_bytes=sum(r['bytes'] for r in listing.values()),
                       excluded_frame_bytes=excluded, data_bytes=sum(r['bytes'] for p,r in listing.items() if Path(p).parts[0]=='data'),
                       dump_bytes=sum(r['bytes'] for p,r in listing.items() if Path(p).parts[0]=='dumps'))
            manifest = {'schema_version': VERSION, 'archive_id': archive_id, 'files': listing, 'directories': directories}
            (catalogue / 'manifests').mkdir(exist_ok=True)
            with gzip.open(catalogue / 'manifests' / (archive_id + '.json.gz.partial'), 'wt') as stream:
                json.dump(manifest, stream)
            (catalogue / 'manifests' / (archive_id + '.json.gz.partial')).replace(catalogue / 'manifests' / (archive_id + '.json.gz'))
            if kind == 'simulation':
                config_path, macro = path / 'config.conf', path / 'macroData.csv'
                if config_path.is_file():
                    config, row['config_version_evidence'] = read_config(config_path)
                    row['config_sha256'] = digest(config_path)
                else:
                    row['errors'].append('missing_config')
                if macro.is_file():
                    row.update(macro_summary(macro))
                    if row['malformed_rows']:
                        row['errors'].append('malformed_macro_rows')
                    if row['macro_changed_during_scan']:
                        row['errors'].append('macro_changed_during_scan')
                    if row['unterminated_macro_line']:
                        row['errors'].append('unterminated_macro_line_requires_review')
                else:
                    row['errors'].append('missing_macro')
                if row.get('macro_rows') == 0:
                    row['errors'].append('no_parseable_macro_rows' if row.get('malformed_rows') else 'header_only_macro')
                if 'startLoad' in config and 'maxLoad' in config and row.get('last_load') is not None:
                    start, target = float(config['startLoad']), float(config['maxLoad'])
                    if target > start and not row.get('nonmonotonic_loads'):
                        row['load_completion_percent'] = 100 * (row['last_load'] - start) / (target - start)
                        row['target_reached'] = row['last_load'] >= target - max(1e-10, abs(target)*1e-10)
                    else:
                        row['completion_note'] = 'Requires protocol-specific interpretation'
            # Check key files against the initial manifest to detect active writes.
            for name in ['config.conf', 'macroData.csv']:
                if name in listing and state(path / name) != listing[name]:
                    row['errors'].append(f'{name}_changed_during_inventory')
        except (OSError, ValueError, UnicodeError, csv.Error) as exc:
            row['errors'].append(f'{type(exc).__name__}: {exc}')
        row.update({'config.'+k:v for k,v in config.items()})
        row['campaign_suggestion'] = Path(relative).parts[0] if len(Path(relative).parts)>1 else config.get('experiment', config.get('scenario', 'unassigned'))
        overrides = annotations.get(relative, {})
        if set(overrides) - {'year', 'campaign', 'comments', 'scientific_quality'}:
            raise ValueError(f'Unsupported annotation fields for {relative}')
        row.update(overrides)
        db.execute('INSERT OR REPLACE INTO units VALUES (?, ?, ?, ?)', (key, archive_id, str(source), json.dumps(row, allow_nan=False)))
        db.commit()
        seen.append(key)
        if index % 25 == 0:
            print(f'Inventoried {index} units: {relative}', flush=True)
    # A rescan updates this source only; other source labels/roots remain intact.
    prefix_rows = db.execute('SELECT source_key, metadata FROM units WHERE source_root=?', (str(source),)).fetchall()
    for key, metadata in prefix_rows:
        if key not in seen and json.loads(metadata)['source_label'] == label:
            db.execute('DELETE FROM units WHERE source_key=?', (key,))
    db.commit()
    export_tables(db, catalogue)
    print(f'Inventory complete: {len(seen)} units. Sources were not changed.', flush=True)


def export_tables(db, catalogue):
    rows = [json.loads(r[0]) for r in db.execute('SELECT metadata FROM units ORDER BY source_key')]
    for filename, selected in [('simulations.csv', [r for r in rows if r['kind']=='simulation']),
                               ('assets.csv', [r for r in rows if r['kind']!='simulation'])]:
        columns = sorted({k for r in selected for k in r if k != 'macro_columns'})
        macro_columns = sorted({c for r in selected for c in r.get('macro_columns', [])})
        columns += ['column.'+c for c in macro_columns]
        with (catalogue / (filename+'.partial')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in selected:
                flat = {k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items() if k!='macro_columns'}
                flat.update({'column.'+c:(c in row['macro_columns']) if 'macro_columns' in row else None for c in macro_columns})
                writer.writerow(flat)
        (catalogue / (filename+'.partial')).replace(catalogue / filename)


def safe_component(value):
    value = str(value)
    if not value or value in {'.', '..'} or '/' in value or '\\' in value:
        raise ValueError(f'Invalid archive path component: {value!r}')
    return value


def destination_for(root, row):
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(row['source_relative_path']).name)[:80]
    return root / safe_component(row['year']) / safe_component(row['campaign']) / (name+'__'+row['archive_id'])


def plan(catalogue, destination):
    if not (catalogue / 'catalogue.sqlite').is_file():
        raise ValueError('Inventory catalogue does not exist')
    db = database(catalogue)
    rows = [json.loads(r[0]) for r in db.execute('SELECT metadata FROM units')]
    entries = [dict(archive_id=r['archive_id'], source=str(Path(r['source_root'])/r['source_relative_path']),
                    destination=str(destination_for(destination, r)), logical_bytes=r.get('logical_bytes',0),
                    status='blocked' if r['active'] or r['errors'] else 'needs_assignment' if r['year']=='unknown' or r['campaign']=='unassigned' else 'ready',
                    errors=r['errors']) for r in rows]
    write_json(catalogue / 'plan.json', {'schema_version':VERSION, 'created_at':stamp(), 'entries':entries})
    summary = {s:sum(r['status']==s for r in entries) for s in ['ready','needs_assignment','blocked']}
    summary['logical_bytes'] = sum(r['logical_bytes'] for r in entries)
    print(json.dumps(summary, indent=2))
    return entries


def validate_archive(path, expected):
    observed = {}
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or member.name in observed:
                raise ValueError(f'Unexpected archive member: {member.name}')
            with archive.extractfile(member) as stream:
                observed[member.name] = stream_digest(stream)
    if observed != expected:
        raise ValueError(f'Archive checksum mismatch: {path}')


def verify(destination):
    metadata = json.loads((destination / 'archive_manifest.json').read_text())
    for name, expected in metadata['copied_files'].items():
        if digest(destination / name) != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    for name, members in metadata['payloads'].items():
        validate_archive(destination / name, members)
    expected_files = set(metadata['copied_files']) | set(metadata['payloads']) | {'archive_manifest.json'}
    actual = {str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()}
    if actual != expected_files:
        raise ValueError(f'Unexpected/missing destination files: {actual ^ expected_files}')
    return metadata


def import_one(catalogue, root, archive_id, year=None, campaign=None, reserve_gb=100):
    if not math.isfinite(reserve_gb) or reserve_gb < 0:
        raise ValueError('Free-space reserve must be finite and non-negative')
    db = database(catalogue)
    result = db.execute('SELECT metadata FROM units WHERE archive_id=?', (archive_id,)).fetchone()
    if result is None:
        raise ValueError(f'Unknown archive ID: {archive_id}')
    row = json.loads(result[0])
    if row['active'] or row['errors']:
        raise ValueError(f'Import blocked: active={row["active"]}, errors={row["errors"]}')
    row.update(year=year or row['year'], campaign=campaign or row['campaign'])
    if row['campaign']=='unassigned':
        raise ValueError('Assign a campaign first; unknown production years may remain unknown')
    source = Path(row['source_root']) / row['source_relative_path']
    root = root.resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError('Import destination must be outside the source')
    with gzip.open(catalogue / 'manifests' / (archive_id+'.json.gz'), 'rt') as stream:
        listing = json.load(stream)
    for name in [*listing['files'], *listing['directories']]:
        p = Path(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError(f'Unsafe manifest path: {name}')
    if 'archive_manifest.json' in listing['files']:
        raise ValueError('Source contains reserved archive_manifest.json filename')
    if files_under(source)[:2] != (listing['files'], listing['directories']):
        raise ValueError('Source changed since inventory; inventory it again')
    destination = destination_for(root, row)
    if destination.exists():
        verified = verify(destination)
        if verified['source_files'] != listing['files'] or verified['archive_id'] != archive_id:
            raise ValueError('Existing archive belongs to another source snapshot')
        print(f'Already imported and verified: {destination}')
        return destination
    # Reserve the entire uncompressed size; do not assume a compression ratio.
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < row['logical_bytes'] + reserve_gb*1e9:
        raise ValueError('Insufficient free space for conservative import plus reserve')
    stage = root / 'staging' / archive_id
    stage.mkdir(parents=True, exist_ok=True)
    copied, payloads, groups = {}, {}, {}
    for directory in listing['directories']:
        top = Path(directory).parts[0]
        if top in PAYLOADS:
            groups.setdefault(top, [])
    for group in groups:
        if group+'.tar.gz' in listing['files']:
            raise ValueError(f'Payload archive name collides with source: {group}.tar.gz')
    for name in listing['files']:
        top = Path(name).parts[0]
        if source.is_dir() and top in PAYLOADS:
            groups.setdefault(top, []).append(name)
        else:
            p = source/name if source.is_dir() else source
            target = stage/name
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = digest(p)
            if not target.is_file() or digest(target) != expected:
                temporary = target.with_name(target.name+'.partial')
                shutil.copy2(p, temporary)
                if digest(temporary) != expected:
                    raise ValueError(f'Copy checksum mismatch: {p}')
                temporary.replace(target)
            copied[name] = expected
    for directory in listing['directories']:
        if Path(directory).parts[0] not in PAYLOADS:
            (stage / directory).mkdir(parents=True, exist_ok=True)
    for group, names in groups.items():
        name = group+'.tar.gz'
        expected = {p:digest(source/p) for p in names}
        target = stage/name
        if target.exists():
            validate_archive(target, expected)
        else:
            temporary = stage/(name+'.partial')
            with tarfile.open(temporary, 'w:gz', compresslevel=3) as archive:
                for directory in listing['directories']:
                    if Path(directory).parts[0] == group:
                        archive.add(source/directory, arcname=directory, recursive=False)
                for p in names:
                    archive.add(source/p, arcname=p, recursive=False)
            validate_archive(temporary, expected)
            temporary.replace(target)
        payloads[name] = expected
    if files_under(source)[:2] != (listing['files'], listing['directories']):
        raise ValueError('Source changed during import; staging retained, archive not published')
    write_json(stage/'archive_manifest.json', dict(schema_version=VERSION, archive_id=archive_id,
               created_at=stamp(), metadata=row, source_files=listing['files'],
               copied_files=copied, payloads=payloads))
    verify(stage)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(destination)
    print(f'Imported and verified; source retained: {destination}')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    scan = sub.add_parser('inventory')
    scan.add_argument('source', type=Path)
    scan.add_argument('catalogue', type=Path)
    scan.add_argument('--source-label', required=True)
    scan.add_argument('--annotations', type=Path)
    scan.add_argument('--active-run', action='append', default=[])
    prepare = sub.add_parser('plan')
    prepare.add_argument('catalogue', type=Path)
    prepare.add_argument('destination', type=Path)
    copy = sub.add_parser('import')
    copy.add_argument('catalogue', type=Path)
    copy.add_argument('destination', type=Path)
    copy.add_argument('--archive-id', required=True)
    copy.add_argument('--year')
    copy.add_argument('--campaign')
    copy.add_argument('--reserve-gb', type=float, default=100)
    check = sub.add_parser('verify')
    check.add_argument('destination', type=Path)
    export = sub.add_parser('export')
    export.add_argument('catalogue', type=Path)
    export.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command=='inventory':
        annotations = json.loads(args.annotations.read_text()) if args.annotations else None
        inventory(args.source,args.catalogue,args.source_label,annotations,args.active_run)
    elif args.command=='plan':
        plan(args.catalogue,args.destination)
    elif args.command=='import':
        import_one(args.catalogue,args.destination,args.archive_id,args.year,args.campaign,args.reserve_gb)
    elif args.command=='export':
        if not (args.catalogue/'catalogue.sqlite').is_file():
            raise ValueError('Inventory catalogue does not exist')
        args.output.mkdir(parents=True, exist_ok=True)
        export_tables(database(args.catalogue),args.output)
    else:
        verify(args.destination)
        print(f'Verified: {args.destination}')


if __name__=='__main__':
    main()
