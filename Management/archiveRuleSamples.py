#!/usr/bin/env python3
"""Read-only, bounded metadata probes for real archive-rule regression cases."""
import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import zipfile

LIMIT = 128 * 1024


def relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe relative locator: {value!r}')
    return str(path)


def local_probe(root, case):
    base = relative(case['source']['path'])
    records = {}
    with contextlib.ExitStack() as stack:
        archive = stack.enter_context(zipfile.ZipFile(root['path'])) if root['transport']=='zip' else None
        for role, name in case['files'].items():
            if name is None:
                continue
            member = str(PurePosixPath(base) / relative(name))
            if archive:
                info = archive.getinfo(member)
                if info.is_dir():
                    raise ValueError(f'Expected a file: {member}')
                record = {'bytes': info.file_size, 'zip_crc32': info.CRC}
                opener = lambda: archive.open(member)
            else:
                path = Path(root['path']) / member
                if not path.is_file():
                    raise FileNotFoundError(path)
                resolved, parent = path.resolve(), Path(root['path']).resolve()
                if parent not in resolved.parents:
                    raise ValueError(f'Locator escapes its root: {path}')
                record = {'bytes': path.stat().st_size}
                opener = lambda: path.open('rb')
            if role in {'config','macro','historical_audit'}:
                with opener() as stream:
                    data = b''.join(stream.readline(LIMIT+1) for _ in range(3)) if role=='macro' else stream.read(LIMIT+1)
                if len(data)>LIMIT:
                    raise ValueError(f'Metadata probe exceeds {LIMIT} bytes: {member}')
                if role=='macro' and not archive:
                    with opener() as stream:
                        stream.seek(max(0,record['bytes']-8192))
                        tail=stream.read(8192)
                    record.update(tail_text=tail.decode('utf-8',errors='replace'),tail_sha256=hashlib.sha256(tail).hexdigest(),tail_scope='last_8192_bytes_may_start_mid_row')
                    header = data.splitlines()[0].decode('utf-8')
                    tail_rows = tail.decode('utf-8', errors='replace').splitlines()[1:-1]
                    width = len(next(csv.reader([header])))
                    if any(len(next(csv.reader([line]))) != width for line in tail_rows if line and not line.startswith('#')):
                        # A resumed run may have a new #HEADER outside the bounded
                        # tail. Stream CSV metadata only, retaining headers and hash.
                        headers, offset, digest = [], 0, hashlib.sha256()
                        tail_header = header
                        with opener() as stream:
                            for binary in stream:
                                digest.update(binary)
                                if binary.startswith(b'#HEADER:'):
                                    text = binary.decode('utf-8').strip().removeprefix('#HEADER:').strip()
                                    headers.append({'byte_offset': offset, 'text': text})
                                    if offset <= max(0, record['bytes']-8192):
                                        tail_header = text
                                offset += len(binary)
                        record.update(embedded_headers=headers, tail_header_text=tail_header,
                                      schema_scan_sha256=digest.hexdigest(), schema_scan_scope='full_csv_stream_for_headers_only')
                record.update(text=data.decode('utf-8'),sample_sha256=hashlib.sha256(data).hexdigest(),
                              scope='first_3_lines' if role=='macro' else 'entire_file')
            else:
                record['scope']='metadata_only_no_payload_read'
            records[role]=record
    return records


def probe(root, case):
    if root['transport'] not in {'directory','zip','ssh'}:
        raise ValueError(f'Unsupported transport: {root["transport"]}')
    if root['transport']!='ssh':
        return local_probe(root,case)
    host=root['host']
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+',host) or host.startswith('-'):
        raise ValueError(f'Invalid SSH host: {host}')
    request={'root':dict(root,transport='directory'),'case':case}
    command='python3 -c '+shlex.quote(Path(__file__).read_text())+' --serve '+shlex.quote(json.dumps(request))
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=20',host,command],
                          capture_output=True,text=True,timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def validate_records(case, records):
    if 'config' in records:
        match=re.search(r'^#\s*Version:\s*(.*)$',records['config']['text'],re.M)
        marker=match.group(1) if match else ''
        if marker != case['code_evidence']['config_marker']:
            raise ValueError(f'Config version marker changed: {marker!r}')
    for role, expected in case.get('expected_probe_sha256',{}).items():
        if hashlib.sha256(records[role]['text'].encode()).hexdigest()!=expected or records[role]['sample_sha256']!=expected:
            raise ValueError(f'Frozen metadata probe changed: {role}')
    for role, expected in case.get('expected_tail_sha256',{}).items():
        if hashlib.sha256(records[role].get('tail_text','').encode()).hexdigest()!=expected or records[role].get('tail_sha256')!=expected:
            raise ValueError(f'Frozen metadata tail changed: {role}')
    for role, expected in case.get('expected_schema_metadata_sha256',{}).items():
        evidence={k:records[role][k] for k in ('embedded_headers','tail_header_text','schema_scan_sha256')}
        if hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()!=expected:
            raise ValueError(f'Frozen embedded schema evidence changed: {role}')
    return records


def validate_case(root, case):
    return validate_records(case,probe(root,case))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--root',action='append',default=[],metavar='NAME=PATH')
    parser.add_argument('--case',action='append',default=[])
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--snapshot',action='store_true',help='Use frozen metadata, not live sources')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    if manifest.get('schema_version')!=1:
        raise ValueError('Unsupported sample-manifest schema')
    for override in args.root:
        name,path=override.split('=',1)
        manifest['roots'][name]['path']=path
    selected=[c for c in manifest['cases'] if not args.case or c['id'] in args.case]
    if args.case and set(args.case)-{c['id'] for c in selected}:
        raise ValueError('Unknown requested case ID')
    cached={}
    if args.snapshot:
        snapshot=args.manifest.parent/manifest['probe_snapshot_file']
        cached={r['id']:r['files'] for r in json.loads(snapshot.read_text())['results']}
    results=[]
    for case in selected:
        root=manifest['roots'][case['source']['root']]
        item={'id':case['id'],'status':'passed'}
        try:
            if args.snapshot:
                item['files']=validate_records(case,cached[case['id']])
                item['mode']='frozen_snapshot'
            elif args.offline and root['transport']=='ssh':
                item['status']='not_checked_offline'
            else:
                item['files']=validate_case(root,case)
        except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as exc:
            item.update(status='failed',error=str(exc))
        results.append(item)
        print(case['id'],item['status'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'schema_version':1,'results':results},indent=2)+'\n')
    if any(r['status']=='failed' for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--serve':
        request=json.loads(sys.argv[2])
        print(json.dumps(local_probe(request['root'],request['case'])))
    else:
        main()
