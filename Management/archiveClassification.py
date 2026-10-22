#!/usr/bin/env python3
"""Evidence-based, metadata-only archive classification (stdlib; no simulator imports)."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from archiveRuleSamples import validate_records, validate_case

RULE_VERSION = '1'
# Audited diffs, not dates or commit messages. These describe the config checkout,
# never prove which executable produced an output or a resumed segment.
MILESTONES = {
    'element_energy_reference_area': '1f93290',
    'cauchy_calculation_defined': 'cfb09cf',
    'cauchy_update_called': 'f8af5e0',
    'snapshot_threshold_extensive': '672288c',
    'cauchy_march_formula': 'cf71595',
    'cauchy_fixed_reference_formula': 'e5f8c03',
    'stress_area_weighting': '54898f7',
    'transport_setting_exposed': 'deb9688',
}


def parse_config(text):
    values, marker = {}, ''
    for line in text.splitlines():
        match = re.fullmatch(r'#\s*Version:\s*(.*)', line)
        if match:
            marker = match[1].strip()
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        key, sep, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if not sep or not key or key in values:
            raise ValueError(f'Malformed or repeated config key: {line!r}')
        values[key] = value
    if 'scenario' in values:
        if 'experiment' in values and values['experiment'] != values['scenario']:
            raise ValueError('Conflicting scenario and experiment')
        values['experiment'] = values['scenario']
    return values, marker


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Non-finite config number: {value!r}')
    return result


def boolean(value):
    if value.lower() not in {'0', '1', 'true', 'false'}:
        raise ValueError(f'Unexpected boolean: {value!r}')
    return value.lower() in {'1', 'true'}


def git_index(repository, markers):
    def git(*args):
        return subprocess.run(['git', '-C', str(repository), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    fixes = {name: git('rev-parse', ref+'^{commit}') for name, ref in MILESTONES.items()}
    result = {'schema_version': 1, 'rule_version': RULE_VERSION, 'milestones': fixes, 'markers': {}}
    for marker in sorted(set(markers)-{'', 'unknown'}):
        # Only accept git-describe-like identifiers; no arbitrary revision expressions.
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', marker) or marker.startswith('-'):
            raise ValueError(f'Unsafe version marker: {marker!r}')
        ref = marker.removesuffix('-dirty')
        try:
            commit = git('rev-parse', '--verify', ref+'^{commit}')
        except subprocess.CalledProcessError:
            result['markers'][marker] = {'status': 'unresolved'}
            continue
        ancestry = {}
        for name, fix in fixes.items():
            p = subprocess.run(['git', '-C', str(repository), 'merge-base', '--is-ancestor', fix, commit],
                               capture_output=True, text=True)
            if p.returncode not in {0, 1}:
                raise RuntimeError(p.stderr)
            ancestry[name] = p.returncode == 0
        source = git('show', commit+':src/Mesh/tElement.cpp')
        formula = re.search(r'void TElement::m_updateCauchyStress\(\)\s*\{(.*?)\n\}', source, re.S)
        result['markers'][marker] = {
            'status': 'resolved', 'commit': commit,
            'commit_date': git('show', '-s', '--format=%cs', commit),
            'dirty': marker.endswith('-dirty'), 'ancestry': ancestry,
            'cauchy_function': formula[0] if formula else None,
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        }
    return result


def macro_observations(record):
    lines = record['text'].splitlines()
    if not lines:
        raise ValueError('Empty macro CSV')
    header = next(csv.reader([lines[0]]))
    normalized = [re.sub(r'[ .]+', '_', c.strip().lower()).strip('_') for c in header]
    positions = [i for i, c in enumerate(normalized) if c == 'load']
    if len(positions) != 1:
        raise ValueError('Expected exactly one load column')
    load_index = positions[0]
    result = {'columns': header, 'initial_columns': header, 'duplicate_columns': [c for c, n in Counter(header).items() if n > 1],
              'load_column_index': load_index, 'has_total_energy': 'total_energy' in normalized,
              'sigma_columns': [c for c in header if c.lower().startswith('avg_sigma')],
              'piola_columns': [c for c in header if c.lower().startswith('avg_p') and c.lower() in {'avg_p12', 'avg_pxy'}],
              'reversibility_columns': [c for c in header if c.startswith(('rev_', 'is_reversible'))],
              'last_load': None, 'last_sampled_finite_energy_load': None,
              'malformed_sample_rows': 0, 'nonfinite_energy_sample_rows': 0,
              'scan_scope': 'prefix_and_bounded_tail_not_full_csv', 'tail_available': 'tail_text' in record,
              'embedded_headers': record.get('embedded_headers', []),
              'tail_columns': next(csv.reader([record.get('tail_header_text', lines[0])]))}
    # An absent index in the early "Line nr,Load,..." schema is a known layout,
    # not permission to shift arbitrary malformed rows.
    chunks = [(header, lines[1:])]
    result['unterminated_tail'] = False
    if 'tail_text' in record:
        tail = record['tail_text']
        tail_lines = tail.splitlines()
        if record['bytes'] > len(tail.encode()):
            tail_lines = tail_lines[1:]  # may start halfway through a row
        elif tail_lines:
            tail_lines = tail_lines[1:]  # header
        result['unterminated_tail'] = bool(tail) and not tail.endswith('\n')
        if result['unterminated_tail'] and tail_lines:
            tail_lines = tail_lines[:-1]
        chunks.append((next(csv.reader([record.get('tail_header_text', lines[0])])), tail_lines))
    for active_header, chunk in chunks:
        active = [re.sub(r'[ .]+', '_', c.strip().lower()).strip('_') for c in active_header]
        positions = [i for i, c in enumerate(active) if c == 'load']
        if len(positions) != 1:
            raise ValueError('Expected exactly one load column in active schema')
        load_index = positions[0]
        energy_indices = [i for i, c in enumerate(active) if c in {'total_energy', 'avg_energy'}]
        for row in csv.reader(chunk):
            if row and row[0].startswith('#HEADER:'):
                active_header = [row[0].removeprefix('#HEADER:').strip()] + row[1:]
                active = [re.sub(r'[ .]+', '_', c.strip().lower()).strip('_') for c in active_header]
                if active.count('load') != 1:
                    raise ValueError('Expected one load column in embedded header')
                load_index = active.index('load')
                energy_indices = [i for i, c in enumerate(active) if c in {'total_energy', 'avg_energy'}]
                result['tail_columns'] = active_header
                continue
            if not row or row == active_header or row[0].startswith('#'):
                continue
            shift = 1 if active[0] in {'line_nr', 'load_step'} and len(row) == len(active_header)-1 else 0
            if len(row) != len(active_header)-shift:
                result['malformed_sample_rows'] += 1
                continue
            try:
                load = number(row[load_index-shift])
                energies = [float(row[i-shift]) for i in energy_indices]
            except (ValueError, IndexError):
                result['malformed_sample_rows'] += 1
                continue
            result['last_load'] = load
            if energies and not all(math.isfinite(e) for e in energies):
                result['nonfinite_energy_sample_rows'] += 1
            elif energies:
                result['last_sampled_finite_energy_load'] = load
    if result['malformed_sample_rows'] and result['tail_available']:
        # Do not report a prefix load as the final load when the tail is unreadable.
        result['last_load'] = result['last_sampled_finite_energy_load'] = None
    if not result['tail_available']:
        result['last_load'] = result['last_sampled_finite_energy_load'] = None
    result['header_only'] = record['bytes'] == len(record['text'].encode()) and len(lines) == 1
    result['columns'] = list(dict.fromkeys(header + result['tail_columns']))
    result['sigma_columns'] = [c for c in result['columns'] if c.lower().startswith('avg_sigma')]
    result['piola_columns'] = [c for c in result['columns'] if c.lower() in {'avg_p12', 'avg_pxy'}]
    return result


def campaign_candidates(config, source_path):
    """Signatures are candidates, never proof of original job membership."""
    def matches(**signature):
        for key, expected in signature.items():
            if key not in config:
                return False
            actual = config[key]
            if isinstance(expected, (int, float)):
                if not math.isclose(number(actual), expected, rel_tol=1e-10, abs_tol=1e-15):
                    return False
            elif actual != expected:
                return False
        return True
    candidates = []
    def add(name, rule, evidence):
        candidates.append({'name': name, 'rule_id': rule, 'status': 'candidate', 'evidence': evidence})
    if matches(experiment='simpleShear', startLoad=.15, loadIncrement=1e-5, maxLoad=1):
        common = ['experiment=simpleShear', 'startLoad=.15', 'loadIncrement=1e-5', 'maxLoad=1']
        if (matches(minimizer='LBFGS', LBFGSEpsx=1e-6) and
                config.get('meshDiagonal', 'unknown') == 'major' and
                config.get('logDuringMinimization') in {'0', 'false'} and
                config.get('rows') == config.get('cols') and
                number(config.get('rows', '0')) in {50, 100, 150, 200, 250, 300} and
                config.get('reconnectionMethod') in {'none', 'edgeFlip'}):
            add('size_scaling_'+('nonreconnecting' if config['reconnectionMethod']=='none' else 'reconnecting'),
                'campaign.size_scaling.v1', common+['square mesh; LBFGSEpsx=1e-6; major diagonal; no minimization log',
                                               'signature shared with basicJob; original job needs review'])
        if matches(LBFGSEpsg=1e-5, CGEpsg=1e-5, eps=1e-5):
            add('legacy_minimizer_comparison', 'campaign.propper_job.v1', common+['LBFGSEpsg=CGEpsg=eps=1e-5'])
        if matches(rows=100, cols=100, seed=0, minimizer='LBFGS', LBFGSEpsg=1e-8, energyDropThreshold=1e-10):
            add('all_plastic_events', 'campaign.all_events.v1', common+['L=100; seed=0; LBFGSEpsg=1e-8; energyDropThreshold=1e-10'])
    experiment = config.get('experiment', 'unknown')
    if experiment in {'singleDislocationTest', 'doubleDislocationTest', 'reconnectSSTest', 'reversibilityProtocolTest'}:
        add(experiment, 'campaign.experiment_family.v1', ['config experiment='+experiment+'; family only'])
    for token, name in [('stress_repair', 'stress_repair'), ('sigma_rescue', 'stress_repair'),
                        ('delayed_reconnection', 'delayed_reconnection'), ('size_scaling_nonreconnecting', 'size_scaling_nonreconnecting')]:
        if token in source_path.lower() and not any(c['name']==name for c in candidates):
            add(name, 'campaign.path_context.v1', ['source path contains '+token+'; context only'])
    return candidates


def classify(records, source_path, index):
    """No case labels, expected observations, fixture IDs or precomputed hints."""
    config, marker = parse_config(records.get('config', {}).get('text', ''))
    macro = macro_observations(records['macro'])
    review, issues = [], []
    def flag(rule, message):
        review.append({'rule_id': rule, 'reason': message})
    def issue(name, status, evidence, note):
        issues.append({'rule_id': 'science.'+name+'.v1', 'status': status, 'evidence': evidence, 'note': note})
    facts = index['markers'].get(marker, {})
    resolved = facts.get('status') == 'resolved'
    code = {'marker': marker, 'scope': 'config_writing_checkout', 'executable_revision': 'unknown',
            'resumed_segment_revisions': 'unknown', 'era': 'unresolved', 'year': None}
    if resolved:
        code.update(facts)
        code['year'] = int(facts['commit_date'][:4])
        included = [name for name in MILESTONES if facts['ancestry'][name]]
        code['era'] = included[-1] if included else 'before_element_energy_reference_area'
        flag('code.config_scope.v1', 'Recorded version identifies config checkout; executable and resumed segments remain unverified.')
        if facts['dirty']:
            flag('code.dirty.v1', 'Uncommitted changes unavailable; ancestry describes the base revision only.')
    else:
        flag('code.unresolved.v1', 'No resolvable config version; schema is recorded without assigning a calendar year.')
    schema = ('tensor_stress' if 'avg_sigma11' in macro['columns'] else
              'scalar_stress' if macro['sigma_columns'] else
              'legacy_average_energy' if not macro['has_total_energy'] else 'total_and_average_energy')
    code['schema_family'] = schema
    for name in ('element_energy_reference_area', 'stress_area_weighting', 'snapshot_threshold_extensive'):
        included = facts.get('ancestry', {}).get(name)
        status = 'unknown' if included is None else ('present_in_config_checkout' if included else 'absent_from_config_checkout')
        issue(name, status, {'fix_commit': index['milestones'][name], 'config_commit': facts.get('commit'), 'dirty_base_only': facts.get('dirty')}, {
            'element_energy_reference_area': 'Reference-area multiplication and ground-state subtraction changed together; no numerical correction inferred.',
            'stress_area_weighting': 'P uses reference-area and sigma current-area weights after fix. Earlier averages need area-distribution review; energy averaging unchanged.',
            'snapshot_threshold_extensive': 'Controls snapshot selection only: total-drop threshold replaces per-element scaling and plastic-change condition. Not an energy-column correction.',
        }[name])
    if macro['sigma_columns'] or facts.get('cauchy_function'):
        update = facts.get('ancestry', {}).get('cauchy_update_called')
        issue('cauchy_update', 'unknown' if update is None else ('present_in_config_checkout' if update else 'missing_update_in_config_checkout'),
              {'fix_commit': index['milestones']['cauchy_update_called']}, 'Exported column presence does not prove stress was updated correctly.')
        flag('science.cauchy_formula_review.v1', 'Check historical stress formula/reference frame against the executable; later Cauchy commits do not establish universal correctness.')
    method = config.get('reconnectionMethod', 'unknown')
    if 'reconnectingEnabled' in config:
        legacy = 'legacy_enabled_method_unknown' if boolean(config['reconnectingEnabled']) else 'none'
        if method != 'unknown' and (legacy == 'none') != (method == 'none'):
            raise ValueError('Conflicting reconnection settings')
        if method == 'unknown':
            method = legacy
    if method == 'legacy_enabled_method_unknown':
        flag('config.legacy_reconnection.v1', 'Legacy enabled flag does not identify the reconnection algorithm.')
    if method not in {'unknown', 'none'}:
        flag('science.reconnecting_stress.v1', 'Stress reconstruction must use total T=F_e T_p; legacy VTU T fields alone are plastic T_p.')
    transport = boolean(config['minimum_transport_distance']) if 'minimum_transport_distance' in config else None
    campaign = campaign_candidates(config, source_path)
    flag('campaign.membership.v1', 'Candidate signatures/path context need original job-manifest confirmation.' if campaign else 'No campaign signature matched; review source job history.')
    if macro['embedded_headers']:
        flag('code.mixed_schema_segments.v1', 'Embedded CSV headers show schema changes; establish executable provenance for each resumed segment.')
    if macro['duplicate_columns']:
        flag('csv.duplicate_columns.v1', 'Preserve column positions; repeated names cannot be selected unambiguously.')
    if macro['unterminated_tail'] or macro['malformed_sample_rows']:
        flag('csv.truncated_or_malformed.v1', 'Sample includes malformed rows or an unterminated tail; last load excludes incomplete rows.')
    if macro['nonfinite_energy_sample_rows']:
        flag('csv.nonfinite_energy.v1', 'Non-finite energy in sampled rows; reaching the load target does not establish scientific completion.')
    if macro['header_only']:
        flag('csv.header_only.v1', 'No macro observations; retain checkpoint and failure context.')
    progress = {'status': 'unknown', 'percent': None, 'basis': 'last_complete_sampled_load; not whole-file validation'}
    if macro['header_only']:
        progress['status'] = 'no_observations'
    elif macro['last_load'] is not None and all(k in config for k in ('startLoad', 'maxLoad', 'loadIncrement')):
        start, target, step = (number(config[k]) for k in ('startLoad', 'maxLoad', 'loadIncrement'))
        if not step or (target-start)*step <= 0:
            progress['status'] = 'loading_direction_requires_review'
            flag('progress.direction.v1', 'Target and increment directions disagree; no completion percentage inferred.')
        else:
            raw = 100*(macro['last_load']-start)/(target-start)
            progress.update(percent=max(0., min(100., raw)), status='target_reached' if raw>=100-1e-7 else 'partial')
    if macro['nonfinite_energy_sample_rows']:
        progress['scientific_completion'] = 'invalid_energy_in_sample'
    if not macro['tail_available']:
        flag('progress.no_tail.v1', 'ZIP prefix-only probe cannot establish final load or completion.')
    role = 'simulation'
    if 'sigmarescue' in config.get('name', '').lower() or 'stress_repair' in source_path.lower():
        role = 'repair_replay_candidate'
        flag('science.repair_manifest.v1', 'Repair location/name does not certify correctness; need build manifest and parent/segment provenance.')
    if 'historical_audit' in records:
        audit = json.loads(records['historical_audit']['text'])
        role = 'derived_macro_with_historical_analysis'
        if audit.get('archived_population_reproduced_exactly') and 'historical_peak_old_dynamic_mu_diagnostic' in audit:
            issue('historical_analysis', 'historical_only', {'audit_purpose': audit.get('purpose'), 'extraction_revision': audit.get('archived_extraction_revision')},
                  'Associated archived fit uses historical tangent/yield choices. This does not itself invalidate the raw macro CSV; re-extract under current conventions.')
            flag('science.audit_association.v1', 'Verify audit source hashes and exact artifact association before attributing its result to an exported CSV.')
    # These quantities are transition/analysis products, not inferred from energy-like names.
    drops = {name: {'status': 'not_established', 'reason': 'No audited transition semantics or gamma0 analysis manifest in this probe.'}
             for name in ('delta_E_R', 'delta_E_I', 'delta_E_S')}
    flag('science.drop_semantics.v1', 'E_R/E_I/E_S availability and conventions require a dedicated source/extraction audit; no guess from energy column names.')
    return {'rule_version': RULE_VERSION, 'code': code, 'config': config, 'macro': macro,
            'properties': {'experiment': config.get('experiment'), 'reconnection_method': method, 'minimum_transport_distance': transport,
                           'energy_columns': 'total_and_average' if macro['has_total_energy'] else 'average_only_or_legacy', 'role': role},
            'campaign_candidates': campaign, 'progress': progress, 'scientific_issues': issues, 'energy_drops': drops,
            'review': review, 'metadata_status': 'review_required', 'scientific_status': 'not_certified',
            'archive_policy': 'metadata_uncertainty_does_not_block_preservation',
            'integrity_status': 'not_assessed_by_classifier'}


def export(results, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/'classification.json').write_text(json.dumps({'schema_version': 1, 'rule_version': RULE_VERSION, 'results': results}, indent=2)+'\n')
    rows = []
    for r in results:
        row = {'id': r['id'], 'source': r['source'], 'code_year': r['code']['year'], 'code_era': r['code']['era'],
               'version': r['code']['marker'], 'last_load': r['macro']['last_load'], 'last_sampled_finite_energy_load': r['macro']['last_sampled_finite_energy_load'], 'completion_percent': r['progress']['percent'],
               'progress_status': r['progress']['status'], **r['properties'],
               'campaign_candidates': '; '.join(c['name'] for c in r['campaign_candidates']),
               'review_reasons': '; '.join(x['rule_id'] for x in r['review']),
               'archive_policy': r['archive_policy'], 'scientific_status': r['scientific_status']}
        row.update({'config.'+k: v for k, v in r['config'].items()})
        row.update({'issue.'+x['rule_id']: x['status'] for x in r['scientific_issues']})
        row.update({'column.'+c: True for c in r['macro']['columns']})
        rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output/'catalogue.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    review = ['# Classification review queue', '', 'Metadata classification only; no source changes or simulation execution.', '',
              'CSV progress uses bounded tail samples, not a full validation. Version years describe config checkout code.', '']
    for r in results:
        review += ['## '+r['id'], '', f"Source: `{r['source']}`", '',
                   f"Code: {r['code']['marker'] or 'unversioned'}; era: {r['code']['era']}; year: {r['code']['year'] or 'unassigned'}.",
                   f"Campaign candidates: {', '.join(c['name'] for c in r['campaign_candidates']) or 'unassigned'}.",
                   f"Progress: {r['progress']['status']}; last load: {r['macro']['last_load']}; percent: {r['progress']['percent']}.", '']
        review += [f"- `{item['rule_id']}`: {item['reason']}" for item in r['review']]
        review += ['']
    (output/'REVIEW.md').write_text('\n'.join(review))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--live', action='store_true', help='Validate bounded live metadata; default is frozen probes')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--git-repo', type=Path)
    source.add_argument('--code-index', type=Path, help='Previously exported Git evidence for standalone offline use')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported sample manifest')
    cached = {r['id']: r['files'] for r in json.loads((args.manifest.parent/manifest['probe_snapshot_file']).read_text())['results']}
    records = {}
    for case in manifest['cases']:
        records[case['id']] = (validate_case(manifest['roots'][case['source']['root']], case) if args.live
                               else validate_records(case, cached[case['id']]))
    index = (git_index(args.git_repo, [parse_config(r.get('config', {}).get('text', ''))[1] for r in records.values()])
             if args.git_repo else json.loads(args.code_index.read_text()))
    if index['schema_version'] != 1 or index['rule_version'] != RULE_VERSION:
        raise ValueError('Incompatible Git evidence index')
    results = []
    for case in manifest['cases']:
        result = classify(records[case['id']], case['source']['path'], index)
        result.update(id=case['id'], source=case['source'], probe_mode='live' if args.live else 'frozen')
        results.append(result)
    export(results, args.output)
    (args.output/'code_index.json').write_text(json.dumps(index, indent=2)+'\n')
    print(f'Classified {len(results)} cases; {sum(r["code"]["year"] is not None for r in results)} resolved config-code years; {args.output}')


if __name__ == '__main__':
    main()
