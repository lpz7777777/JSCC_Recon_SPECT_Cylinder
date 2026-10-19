"""Read-only catalog boundaries and navigation correctness."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('project_index', ROOT/'tools/project_index.py')
INDEX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INDEX)


class ProjectIndexTests(unittest.TestCase):
    def test_real_catalog_has_valid_required_paths(self):
        result = INDEX.check(INDEX.load(INDEX.CATALOG), INDEX.load(INDEX.SOURCES),
                             INDEX.load(INDEX.DIRECTORIES))
        self.assertTrue(result['passed'], result['errors'])
        self.assertGreaterEqual(result['experiments'], 33)
        self.assertGreater(result['indexed_sources'], 1000)
        self.assertGreater(result['indexed_directories'], 50)

    def test_duplicate_ids_rejected(self):
        item = dict(id='same', entries=[])
        result = INDEX.check(dict(experiments=[item, item], components=[]), dict(files=[]))
        self.assertFalse(result['passed'])

    def test_catalog_cannot_escape_project(self):
        for path in ['../outside.json', str(ROOT/'README.md')]:
            with self.assertRaises(ValueError):
                INDEX.relative_path(path)

    def test_local_only_data_absence_is_not_source_error(self):
        catalog = dict(experiments=[dict(id='local', entries=['README.md'],
                       local_data=['generated-not-present-example/image.float32'])], components=[])
        result = INDEX.check(catalog, dict(files=[]), dict(directories=[
            dict(path='generated-not-present-example', local_only=True)]))
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['local_only_missing']), 2)

    def test_search_case_insensitive_and_chinese(self):
        item = dict(title='Derenzo 单光子218', topic='JSCC')
        self.assertTrue(INDEX.matches(item, 'derenzo'))
        self.assertTrue(INDEX.matches(item, '单光子'))
        self.assertFalse(INDEX.matches(item, 'Poisson'))

    def test_query_does_not_import_production_or_network(self):
        for name in ['paramiko', 'ehe_5e9_workflow', 'derenzo_campaign_workflow',
                     'jscc_5e10_5090_production_queue']:
            self.assertNotIn(name, sys.modules)


if __name__ == '__main__':
    unittest.main()
