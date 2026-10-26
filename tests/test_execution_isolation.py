"""Execution isolation must hold even when users edit a study after one run."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import runner


class WorkspaceTests(unittest.TestCase):
    def test_snapshot_isolation_and_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);study=repo/'experiments/example';study.mkdir(parents=True)
            source=study/'code.py';source.write_text('value=1\n')
            saved=study/'saved.json';saved.write_text('{"count":10}')
            (study/'README.md').write_text('# Example')
            (study/'experiment.json').write_text('{}')
            layout=repo/'layout.json'
            layout.write_text(json.dumps({'files':[
                {'path':'experiments/example/code.py','runtime_path':'study/code.py','owner':1},
                {'path':'experiments/example/saved.json','runtime_path':'study/results/saved.json','owner':1}],
                'runtime_documents':[]}))
            with patch.object(runner,'REPO',repo),patch.object(runner,'LAYOUT',layout):
                workspace=runner.initialize(study,{'id':1},'first',repo/'assets')
                self.assertFalse((workspace/'workspace/study/results/saved.json').exists())
                executable=workspace/'workspace/study/code.py'
                executable.write_text('value=2\n')
                self.assertEqual(source.read_text(),'value=1\n')
                self.assertEqual(saved.read_text(),'{"count":10}')
                source.write_text('value=3\n')
                with self.assertRaisesRegex(ValueError,'Sources changed'):
                    runner.initialize(study,{'id':1},'first',repo/'assets')
                other=runner.initialize(study,{'id':1},'second',repo/'assets')
                self.assertEqual((other/'workspace/study/code.py').read_text(),'value=3\n')


if __name__=='__main__':unittest.main()
