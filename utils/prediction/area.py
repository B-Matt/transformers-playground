import cv2

import numpy as np

from typing import List


def calc_area(img: np.array, treshold_area: float) -> List:
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours: List = []

    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area > treshold_area:
            final_contours.append([cnt, cnt_area])

    final_contours = sorted(final_contours, key = lambda x:x[1], reverse=True)
    return final_contours

def calc_biggest_area(img: np.array, treshold_area: float) -> List:
    return calc_area(img, treshold_area)[0]

def calc_mean_area(img: np.array, treshold_area: float):
    contours = calc_area(img, treshold_area)
    if len(contours) <= 0:
        return 0, 0.0

    cnt_area_list = [area[1] for area in contours]
    return len(contours), np.mean(cnt_area_list)