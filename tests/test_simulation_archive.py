import contextlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from Management import simulationArchive as archive


class SimulationArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source, self.catalogue, self.destination = [self.base / n for n in ['source','catalogue','archive']]
        self.run = self.source/'campaign/run'
        self.run.mkdir(parents=True)
        (self.run/'config.conf').write_text('# Version: abc123-dirty\nstartLoad = 0.5\nmaxLoad = 1.0\n')
        (self.run/'macroData.csv').write_text('load_step,load,total_energy\n0,0.5,2\n1,1,1\n')
        for name, value in [('data/a.vtu','mesh'),('dumps/restart.xml.gz','restart'),('frames/frame000.png','frame'),('frames/movie.mp4','video')]:
            p=self.run/name
            p.parent.mkdir(exist_ok=True)
            p.write_text(value)
        (self.run/'data/empty').mkdir()

    def scan(self, active=()):
        with contextlib.redirect_stdout(io.StringIO()):
            archive.inventory(self.source,self.catalogue,'test',active=active)
        with sqlite3.connect(self.catalogue/'catalogue.sqlite') as db:
            return json.loads(db.execute("SELECT metadata FROM units WHERE metadata LIKE '%simulation%'").fetchone()[0])

    def copy(self,row):
        with contextlib.redirect_stdout(io.StringIO()):
            return archive.import_one(self.catalogue,self.destination,row['archive_id'],campaign='practice',reserve_gb=0)

    def test_nested_discovery_includes_loose_archives(self):
        (self.source/'old.zip').write_text('zip')
        units=list(archive.discover(self.source))
        self.assertEqual({kind for _,kind in units},{'simulation','asset'})

    def test_completion_uses_start_load_and_no_config_defaults(self):
        (self.run/'macroData.csv').write_text('load_step,load\n1,0.75\n')
        row=self.scan()
        self.assertEqual(row['load_completion_percent'],50)
        self.assertEqual(row['config_version_evidence'],'abc123-dirty')
        self.assertEqual(row['executable_version'],'unknown')
        self.assertNotIn('config.seed',row)

    def test_legacy_missing_index_and_repeated_header_are_explicit(self):
        p=self.run/'macroData.csv'
        p.write_text('load_step,load,energy\n0,0.5,1\n#HEADER:load_step,load,energy\n0.75,2\nload_step,load,energy\n2,1,3\n')
        row=self.scan()
        self.assertEqual(row['macro_rows'],3)
        self.assertEqual(row['legacy_missing_index_rows'],1)

    def test_malformed_data_blocks_import(self):
        (self.run/'macroData.csv').write_text('load_step,load\n0,0.5\nbad,row,extra\n')
        row=self.scan()
        self.assertIn('malformed_macro_rows',row['errors'])
        with self.assertRaisesRegex(ValueError,'Import blocked'):
            self.copy(row)

    def test_duplicate_header_still_preserves_file_inventory(self):
        (self.run/'macroData.csv').write_text('load,load\n0.5,0.5\n')
        row=self.scan()
        self.assertTrue(row['errors'])
        self.assertGreater(row['logical_bytes'],0)
        self.assertTrue((self.catalogue/'manifests'/(row['archive_id']+'.json.gz')).is_file())

    def test_unterminated_row_cannot_be_interpreted_as_legacy_missing_index(self):
        (self.run/'macroData.csv').write_text('load_step,load,total_energy\n0,0.5,2\n1,1')
        row=self.scan()
        self.assertIn('unterminated_macro_line_requires_review',row['errors'])

    def test_active_and_changed_sources_are_blocked(self):
        row=self.scan(active=['campaign/run'])
        with self.assertRaisesRegex(ValueError,'Import blocked'):
            self.copy(row)
        row=self.scan()
        (self.run/'data/a.vtu').write_text('changed mesh')
        with self.assertRaisesRegex(ValueError,'Source changed'):
            self.copy(row)

    def test_archive_roundtrip_preserves_videos_excludes_frames_and_is_idempotent(self):
        before={str(p.relative_to(self.run)):archive.digest(p) for p in self.run.rglob('*') if p.is_file()}
        row=self.scan()
        target=self.copy(row)
        self.assertEqual(target,self.copy(row))
        result=archive.verify(target)
        self.assertEqual((target/'frames/movie.mp4').read_text(),'video')
        self.assertFalse((target/'frames/frame000.png').exists())
        self.assertEqual(set(result['payloads']),{'data.tar.gz','dumps.tar.gz'})
        self.assertEqual(before,{str(p.relative_to(self.run)):archive.digest(p) for p in self.run.rglob('*') if p.is_file()})
        self.assertEqual(row['archive_id'],self.scan()['archive_id'])

    def test_corrupt_archive_is_detected(self):
        target=self.copy(self.scan())
        (target/'data.tar.gz').write_bytes(b'broken')
        with self.assertRaises(Exception):
            archive.verify(target)

    def test_interrupted_staging_resumes_without_publishing_partial_archive(self):
        row=self.scan()
        stage=self.destination/'staging'/row['archive_id']
        stage.mkdir(parents=True)
        shutil.copy2(self.run/'macroData.csv',stage/'macroData.csv')
        (stage/'data.tar.gz.partial').write_bytes(b'interrupted stream')
        target=self.copy(row)
        archive.verify(target)
        self.assertFalse(stage.exists())

    def test_free_space_guard_prevents_creating_staging(self):
        row=self.scan()
        with patch.object(archive.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(100,100,0)):
            with self.assertRaisesRegex(ValueError,'Insufficient free space'):
                self.copy(row)
        self.assertFalse(self.destination.exists())

    def test_standalone_copy_has_no_project_import_dependency(self):
        script=self.base/'simulationArchive.py'
        shutil.copy2(archive.__file__,script)
        result=subprocess.run([sys.executable,'-I',str(script),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':
    unittest.main()
