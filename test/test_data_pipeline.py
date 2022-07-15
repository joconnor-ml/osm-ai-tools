"""Quick integration test of entire data pipeline"""
from osm_ai_tools.scripts import data_pipeline
import unittest


class TestPipeline(unittest.TestCase):
    def test_pipeline(self):
        data_pipeline.main("test/config.json")
