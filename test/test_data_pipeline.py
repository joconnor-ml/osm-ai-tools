"""Quick integration test of entire data pipeline"""
from osm_ai_tools.scripts import data_pipeline
import unittest
import os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

class TestPipeline(unittest.TestCase):
    def test_pipeline(self):
        data_pipeline.main(CONFIG_FILE)