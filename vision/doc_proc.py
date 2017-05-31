#Simple OpenCV-based Q-card processor for Haiko
#Plant 138 - Patrick Weber

import cv2
import numpy as np

qualc = cv2.imread('q_card.tif')

class CardProcessor():
    def __init__(self):
        self.quality_card = cv2.imread('q_card.jpg')
        self.original_quality_card = self.quality_card.copy()

        self.detect_contours_on_image()
        self.detect_stamp_fields__from_aspect_ratio_and_size()
        cv2.drawContours(self.quality_card, self.correct_ratio_contours, -1, (0,255,0), 3)

        cv2.imshow("Quality Card", self.cleaned_up_image())
        cv2.imshow("Canny Edge Detection", self.edged)
        cv2.imwrite("processed_quality_card.jpg", self.quality_card)
        cv2.imshow(self.h)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cleaned_up_image(self):
        # convert to gray scale and blur the noise out
        HSV = cv2.cvtColor(self.quality_card, cv2.COLOR_BGR2HSV)
        self.h, s, v = cv2.split(HSV)

      #  blurred = cv2.GaussianBlur(gray, (1,1), 0)

        return self.original_quality_card

    def detect_contours_on_image(self):
        contours_edges = cv2.Canny(self.cleaned_up_image(), 75, 200)
        #_, self.thresh = cv2.threshold(self.cleaned_up_image(), 127,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        self.edged = contours_edges
        (_, self.contours, _) = cv2.findContours(self.edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        self.contours = sorted(self.contours, key=cv2.contourArea, reverse=True)

    def detect_stamp_fields__from_aspect_ratio_and_size(self):
        # filter out boxes based on area
        correct_size_contours = []

        for cnt in self.contours:
            x,y,w,h = cv2.boundingRect(cnt)
            area = w*h
            if area > 1000 and area < 1000000000:
                correct_size_contours.append(cnt)

        self.correct_size_contours = correct_size_contours


        # filter out boxes based on aspect ratio
        correct_ratio_contours = []

        for cnt in self.correct_size_contours:
            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            if aspect_ratio >= 1.3 and aspect_ratio < 1.9:
                correct_ratio_contours.append(cnt)

        self.correct_ratio_contours = correct_ratio_contours


if __name__ == "__main__":
    henry = CardProcessor()
