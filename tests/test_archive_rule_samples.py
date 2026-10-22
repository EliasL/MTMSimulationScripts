import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SCRIPT=Path(__file__).with_name('archiveRuleSamples.py')
if not SCRIPT.exists():
    SCRIPT=Path(__file__).resolve().parents[1]/'Management/archiveRuleSamples.py'
spec=importlib.util.spec_from_file_location('archive_rule_samples',SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ArchiveRuleSampleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'run').mkdir()
        (self.root/'run/config.conf').write_text('# Version: abc\nseed=0\n')
        (self.root/'run/macroData.csv').write_text('load,load\n0,1\n')
        self.case={'source':{'path':'run'},'files':{'config':'config.conf','macro':'macroData.csv','mesh':None},'code_evidence':{'config_marker':'abc'}}
        self.source={'transport':'directory','path':str(self.root)}

    def test_unsafe_member_paths_are_rejected(self):
        for name in ['../secret','/absolute','run/../../secret']:
            with self.assertRaises(ValueError):module.relative(name)

    def test_directory_probe_keeps_duplicate_headers_and_tail(self):
        r=module.validate_case(self.source,self.case)
        self.assertEqual(r['macro']['text'],'load,load\n0,1\n')
        self.assertIn('tail_sha256',r['macro'])

    def test_zip_probe_reads_selected_members_without_extraction(self):
        path=self.root/'runs.zip'
        with zipfile.ZipFile(path,'w') as z:
            for p in (self.root/'run').iterdir():z.write(p,str(p.relative_to(self.root)))
        before=set(self.root.rglob('*'))
        r=module.validate_case({'transport':'zip','path':str(path)},self.case)
        self.assertEqual(r['config']['text'],'# Version: abc\nseed=0\n')
        self.assertEqual(before,set(self.root.rglob('*')))

    def test_frozen_probe_detects_config_changes_without_a_version_change(self):
        r=module.validate_case(self.source,self.case)
        self.case['expected_probe_sha256']={'config':r['config']['sample_sha256']}
        (self.root/'run/config.conf').write_text('# Version: abc\nseed=1\n')
        with self.assertRaisesRegex(ValueError,'Frozen metadata probe changed'):module.validate_case(self.source,self.case)

    def test_frozen_tail_detects_changes_beyond_the_header(self):
        r=module.validate_case(self.source,self.case)
        self.case['expected_tail_sha256']={'macro':r['macro']['tail_sha256']}
        (self.root/'run/macroData.csv').write_text('load,load\n0,2\n')
        with self.assertRaisesRegex(ValueError,'Frozen metadata tail changed'):module.validate_case(self.source,self.case)

    def test_large_configuration_is_not_read_unbounded(self):
        (self.root/'run/config.conf').write_bytes(b'x'*(module.LIMIT+1))
        with self.assertRaisesRegex(ValueError,'exceeds'):module.validate_case(self.source,self.case)

    def test_missing_config_can_be_an_explicit_case(self):
        self.case['files']['config']=None
        (self.root/'run/config.conf').unlink()
        r=module.validate_case(self.source,self.case)
        self.assertNotIn('config',r)


    def test_changed_tail_width_streams_embedded_schema_without_payloads(self):
        path=self.root/'run/macroData.csv'
        path.write_text('load,avg_energy\n'+('0,1\n'*3000)+'#HEADER:load,avg_energy,extra\n'+('1,2,3\n'*3000))
        r=module.local_probe(self.source,self.case)['macro']
        self.assertEqual(len(r['embedded_headers']),1)
        self.assertEqual(r['tail_header_text'],'load,avg_energy,extra')
        self.assertEqual(r['schema_scan_scope'],'full_csv_stream_for_headers_only')

    def test_equal_width_tail_keeps_bounded_probe(self):
        r=module.local_probe(self.source,self.case)['macro']
        self.assertNotIn('schema_scan_sha256',r)

    def test_all_frozen_real_cases_match_manifest(self):
        directory=Path(__file__).parent
        if not (directory/'archive_rule_samples.json').exists():
            directory=directory/'fixtures'
        manifest=json.loads((directory/'archive_rule_samples.json').read_text())
        probes=json.loads((directory/manifest['probe_snapshot_file']).read_text())
        by_id={r['id']:r['files'] for r in probes['results']}
        self.assertEqual(len(manifest['cases']),20)
        self.assertEqual(set(by_id),{c['id'] for c in manifest['cases']})
        for case in manifest['cases']:
            module.validate_records(case,by_id[case['id']])


if __name__=='__main__':unittest.main()
