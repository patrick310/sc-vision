# -*- coding: utf-8 -*-

import unittest
from vdp.vision_data_processor import VisionDataProcessor
from vdp.utils import create_model_config


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_absolute_truth_and_meaning(self):
        assert True

    def test_build_and_fit_model(self):
        dataprocessor = VisionDataProcessor()
        dataprocessor.create_simple_categorical_model()
        dataprocessor.fit_model()


if __name__ == '__main__':
    unittest.main()