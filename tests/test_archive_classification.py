import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent/'Management').is_dir() else HERE
SCRIPT = ROOT/'Management/archiveClassification.py'
if not SCRIPT.exists():
    SCRIPT = HERE/'archiveClassification.py'
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location('archive_classification', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
FIXTURES = HERE/'fixtures' if (HERE/'fixtures').is_dir() else HERE


class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((FIXTURES/'archive_rule_samples.json').read_text())
        cls.probes = {r['id']: r['files'] for r in json.loads((FIXTURES/'archive_rule_probes.json').read_text())['results']}
        cls.index = json.loads((FIXTURES/'archive_rule_code_index.json').read_text())
        cls.results = [m.classify(cls.probes[c['id']], c['source']['path'], cls.index) for c in cls.manifest['cases']]

    def test_all_twenty_real_cases_have_expected_code_year_and_progress(self):
        years = [None,None,None,2025,2025,2025,2025,2026,2026,2026,2026,2026,2026,2026,2026,2026,None,None,2026,None]
        states = ['partial','unknown','unknown','partial','loading_direction_requires_review','target_reached',
                  'target_reached','target_reached','target_reached','target_reached','partial','target_reached',
                  'partial','target_reached','target_reached','target_reached','partial','target_reached','no_observations','unknown']
        self.assertEqual(len(self.results),20)
        for case, result, year, state in zip(self.manifest['cases'], self.results, years, states):
            with self.subTest(case=case['id']):
                self.assertEqual(result['code']['year'],year)
                self.assertEqual(result['progress']['status'],state)
                self.assertTrue(result['review'])
                self.assertEqual(result['scientific_status'],'not_certified')

    def test_code_years_are_config_checkout_only(self):
        for r in self.results:
            self.assertEqual(r['code']['executable_revision'],'unknown')
            self.assertEqual(r['code']['scope'],'config_writing_checkout')
        self.assertTrue(self.results[3]['code']['dirty'])
        self.assertIn('code.dirty.v1',[x['rule_id'] for x in self.results[3]['review']])

    def test_no_calendar_guess_from_paths_or_filenames(self):
        r=m.classify(self.probes['17_partial_size_scaling'],'2023/2026/definitely_fixed',self.index)
        self.assertIsNone(r['code']['year'])
        self.assertEqual(r['code']['era'],'unresolved')

    def test_campaign_candidates_match_science_and_exclude_variants(self):
        expected={0:'legacy_minimizer_comparison',3:'all_plastic_events',4:'singleDislocationTest',
                  5:'legacy_minimizer_comparison',6:'doubleDislocationTest',8:'size_scaling_nonreconnecting',
                  9:'size_scaling_nonreconnecting',11:'reconnectSSTest',12:'reversibilityProtocolTest',
                  14:'doubleDislocationTest',15:'doubleDislocationTest',16:'size_scaling_nonreconnecting',
                  17:'stress_repair',18:'delayed_reconnection'}
        for i,name in expected.items():
            with self.subTest(case=i):
                self.assertIn(name,[c['name'] for c in self.results[i]['campaign_candidates']])
        for i in [1,2,7,10,13,19]:
            self.assertEqual(self.results[i]['campaign_candidates'],[])
        for r in self.results:
            self.assertTrue(all(c['status']=='candidate' for c in r['campaign_candidates']))

    def test_transport_pair_is_distinguished_by_explicit_settings(self):
        self.assertEqual(self.results[14]['code'],self.results[15]['code'])
        self.assertIs(self.results[14]['properties']['minimum_transport_distance'],False)
        self.assertIs(self.results[15]['properties']['minimum_transport_distance'],True)

    def test_boolean_reconnection_does_not_invent_edgeflip(self):
        self.assertEqual(self.results[6]['properties']['reconnection_method'],'legacy_enabled_method_unknown')

    def test_fix_ancestry_keeps_energy_and_snapshot_rules_separate(self):
        def status(i,key):
            return next(x['status'] for x in self.results[i]['scientific_issues'] if x['rule_id']=='science.'+key+'.v1')
        self.assertEqual(status(8,'snapshot_threshold_extensive'),'absent_from_config_checkout')
        self.assertEqual(status(9,'snapshot_threshold_extensive'),'present_in_config_checkout')
        self.assertEqual(status(12,'stress_area_weighting'),'absent_from_config_checkout')
        self.assertEqual(status(13,'stress_area_weighting'),'present_in_config_checkout')
        self.assertEqual(status(8,'element_energy_reference_area'),status(9,'element_energy_reference_area'))
        self.assertEqual(status(16,'element_energy_reference_area'),'unknown')

    def test_duplicate_header_positions_are_preserved(self):
        r=self.results[0]['macro']
        self.assertIn('Nr CG iterations',r['duplicate_columns'])
        self.assertEqual(r['initial_columns'].count('Nr CG iterations'),2)
        self.assertEqual(r['load_column_index'],0)

    def test_resumed_schema_uses_actual_embedded_header(self):
        r=self.results[16]
        self.assertEqual(len(r['macro']['embedded_headers']),1)
        self.assertEqual(r['macro']['last_load'],.60454)
        self.assertAlmostEqual(r['progress']['percent'],53.4752941176)
        self.assertIn('avg_P12',r['macro']['tail_columns'])
        self.assertIn('code.mixed_schema_segments.v1',[x['rule_id'] for x in r['review']])

    def test_unrecognized_tail_schema_never_reports_prefix_as_final_load(self):
        record=copy.deepcopy(self.probes['17_partial_size_scaling']['macro'])
        record.pop('tail_header_text')
        record.pop('embedded_headers')
        r=m.macro_observations(record)
        self.assertIsNone(r['last_load'])
        self.assertGreater(r['malformed_sample_rows'],0)

    def test_truncated_tail_does_not_contribute_partial_row(self):
        r=self.results[10]['macro']
        self.assertTrue(r['unterminated_tail'])
        self.assertEqual(r['last_load'],.15584)

    def test_target_reached_with_nan_energy_is_flagged(self):
        r=self.results[6]
        self.assertEqual(r['progress']['percent'],100)
        self.assertGreater(r['macro']['nonfinite_energy_sample_rows'],0)
        self.assertEqual(r['progress']['scientific_completion'],'invalid_energy_in_sample')

    def test_archive_policy_keeps_failed_partial_and_unknown_runs(self):
        for i in [1,3,6,10,16,18]:
            self.assertEqual(self.results[i]['archive_policy'],'metadata_uncertainty_does_not_block_preservation')
            self.assertEqual(self.results[i]['integrity_status'],'not_assessed_by_classifier')
        self.assertTrue(self.results[18]['macro']['header_only'])

    def test_repair_path_does_not_certify_science(self):
        r=self.results[17]
        self.assertEqual(r['properties']['role'],'repair_replay_candidate')
        self.assertTrue(all(d['status']=='not_established' for d in r['energy_drops'].values()))

    def test_historical_analysis_issue_is_artifact_scoped(self):
        r=self.results[19]
        issue=next(x for x in r['scientific_issues'] if x['rule_id']=='science.historical_analysis.v1')
        self.assertEqual(issue['status'],'historical_only')
        self.assertIn('does not itself invalidate the raw macro CSV',issue['note'])

    def test_no_test_label_or_expected_observation_enters_classifier(self):
        for c,r in zip(self.manifest['cases'],self.results):
            changed=dict(c,id='fake',label='all correct',coverage=[],expected_observations={'fixed':True},code_evidence={})
            self.assertEqual(m.classify(self.probes[c['id']],changed['source']['path'],self.index),r)

    def test_conflicting_configs_and_duplicate_load_fail_loudly(self):
        for text in ['a=1\na=2','scenario=a\nexperiment=b','garbage']:
            with self.assertRaises(ValueError): m.parse_config(text)
        with self.assertRaises(ValueError):
            m.macro_observations({'text':'load,load\n0,1\n','bytes':14})

    def test_embedded_header_inside_tail_is_applied(self):
        text='load,avg_energy\n0,1\n#HEADER:load,avg_energy,extra\n1,2,3\n'
        r=m.macro_observations({'text':text,'tail_text':text,'bytes':len(text)})
        self.assertEqual(r['last_load'],1)
        self.assertEqual(r['malformed_sample_rows'],0)

    def test_export_has_one_row_per_source_and_wide_columns(self):
        import csv
        results=[dict(r,id=c['id'],source=c['source']) for c,r in zip(self.manifest['cases'],self.results)]
        with tempfile.TemporaryDirectory() as folder:
            m.export(results,Path(folder))
            with (Path(folder)/'catalogue.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),20)
            self.assertGreater(len(rows[0]),100)


if __name__=='__main__': unittest.main()
