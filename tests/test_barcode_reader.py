import unittest
import cv2
from production_code_reader.read_from_image import read_production_number_from_image
from production_code_reader.read_from_image import is_label_in_image
from production_code_reader.read_from_image import locate_label_in_image

negative_image = 'negative.jpg'
positive_image = 'positive_barcode.jpg'


class BarcodeTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_check_images(self):
        self.assertIsNotNone(cv2.imread(negative_image))
        self.assertIsNotNone(cv2.imread(positive_image))

    def test_returns_none_if_no_label(self):
        self.assertIsNone(read_production_number_from_image(negative_image))

    def test_returns_number_from_barcode(self):
        self.assertIsNotNone(read_production_number_from_image(positive_image))

    def test_can_find_barcode_in_image(self):
        self.assertIsNotNone(is_label_in_image(cv2.imread(positive_image)))


if __name__ == '__main__':
    unittest.main()

