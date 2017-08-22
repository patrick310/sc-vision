# -*- coding: utf-8 -*-

from lvp.lean_vision_processor import LeanVisionProcessor

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def setUp(self):
        self.lvp = LeanVisionProcessor()
        self.lvp.set_test_case()

    def test_lvp_initialization(self):
        assert self.lvp.current_state is "Initialized"

    def test_model_resolution(self):
        self.assertEqual(self.lvp.get_model_resolution(),(224, 224))

    def test_can_detect_an_elephant(self):
        self.elephant_predictions = self.lvp.predict_top_class(self.lvp.image_from_file('elephant.testjpg'))
        self.assertIn('African_elephant', self.elephant_predictions)

    def test_can_set_and_trigger_alarm(self):
        self.lvp.set_alarm_case('African_elephant')
        self.assertTrue(self.lvp.has_alarm
                        (self.lvp.predict_top_class(self.lvp.image_from_file('elephant.testjpg'))))

    def test_can_identify_save_flags(self):
        self.lvp.set_file_save_case('African_elephant')
        self.assertTrue(self.lvp.has_save_flag(
            self.lvp.predict_top_class(self.lvp.image_from_file('elephant.testjpg'))))

    def test_model_saves_files_with_flags(self):
        self.lvp.set_file_save_case('African_elephant')


if __name__ == '__main__':
    unittest.main()
