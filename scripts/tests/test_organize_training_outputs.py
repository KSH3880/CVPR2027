"""CPU checks for output relocation and protection of live paths."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('organize_outputs', Path(__file__).resolve().parents[1] / 'organize_training_outputs.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


class MigrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'runs/experiment'
        self.destination = self.root / 'project/output/experiment'
        (self.source / 'nn').mkdir(parents=True)
        (self.source / 'nn/model.pth').write_bytes(b'checkpoint')
        (self.source / 'latest.pth').symlink_to('nn/model.pth')
        self.manifest = self.root / 'manifest.json'

    def run_migration(self, active=()):
        with patch.object(migration, 'ROOT', self.root), patch.object(migration, 'plan', return_value=[(self.source, self.destination)]), patch.object(migration, 'active_references', return_value=active), patch.object(sys, 'argv', ['organize', '--apply', '--manifest', str(self.manifest)]):
            migration.main()

    def test_preserves_checkpoint_inode_and_old_paths(self):
        inode = (self.source / 'nn/model.pth').stat().st_ino
        self.run_migration()
        self.assertTrue(self.source.is_symlink())
        self.assertEqual((self.destination / 'nn/model.pth').stat().st_ino, inode)
        self.assertEqual((self.destination / 'latest.pth').read_bytes(), b'checkpoint')
        self.run_migration()  # Idempotent; preserves the first manifest.
        self.assertEqual(len(json.loads(self.manifest.read_text())), 1)

    def test_live_writer_stays_in_place(self):
        self.run_migration([(42, [str(self.source / 'nn/model.pth')])])
        self.assertFalse(self.source.is_symlink())
        self.assertTrue(self.destination.is_symlink())
        self.assertEqual(self.destination.resolve(), self.source)

    def test_collision_never_changes_source(self):
        self.destination.mkdir(parents=True)
        with self.assertRaises(RuntimeError):
            self.run_migration()
        self.assertFalse(self.source.is_symlink())
        self.assertTrue((self.source / 'nn/model.pth').exists())

    def test_external_relative_link_keeps_target(self):
        target = self.root / 'runs/reference.pth'
        target.write_bytes(b'reference')
        (self.source / 'reference.pth').symlink_to('../reference.pth')
        self.run_migration()
        self.assertEqual((self.destination / 'reference.pth').resolve(), target)


if __name__ == '__main__':
    unittest.main()
