"""Run with .venv/bin/python -m unittest discover -s scripts/baseline -p 'test_*.py'."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from data import grade, revise, summarize


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(id=0, ranks=['4', '3', '2', 'K'], numbers=[4, 3, 2, 10], solutions=['4*3+2+10'])

    def answer(self, formula):
        return json.dumps(dict(cards=self.case['ranks'], number=self.case['numbers'], formula=formula))

    def test_whitespace(self):
        for formula in ['4*3+2+10', ' 4 * 3 + 2 + 10 = 24 ', '(4 *\n 3) + 2 + 10', '4\t*3+2+10']:
            self.assertTrue(grade(self.case, self.answer(formula), 'vision')['success'])

    def test_reject_invalid_arithmetic(self):
        for formula in ['4*3+2+1 0', '4*3+2+2', '4**3+2+10', '4*3+2+True', '4/(3-3)+10', '__import__("os")']:
            self.assertFalse(grade(self.case, self.answer(formula), 'language')['success'])

    def test_recognition_still_required(self):
        answer = self.answer('4 * 3 + 2 + 10').replace('"K"', '"Q"')
        self.assertFalse(grade(self.case, answer, 'vision')['success'])
        self.assertTrue(grade(self.case, answer, 'language')['success'])

    def test_original_summary_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            original='{"complete":true}'
            (folder/'summary.json').write_text(original)
            with self.assertRaises(ValueError):summarize(folder)
            self.assertEqual((folder/'summary.json').read_text(),original)

    def test_labeled_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'manifest.json').write_text('{}')
            (folder/'cases.jsonl').write_text(json.dumps(self.case)+'\n')
            (folder/'responses.jsonl').write_text(json.dumps(dict(id=0,mode='language',response=self.answer('4 * 3 + 2 + 10')))+'\n')
            with contextlib.redirect_stdout(io.StringIO()):summarize(folder,'whitespace-v2')
            summary=json.loads((folder/'summary.whitespace-v2.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['modes']['language']['success_rate'],1)
            self.assertEqual(summary['scoring_version'],'whitespace-v2')
            self.assertFalse((folder/'summary.json').exists())


if __name__=='__main__':unittest.main()
