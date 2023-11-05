import cv2 as cv
import torch
from bfal.utils import (
    get_distance_of_2_points,
    midpoint,
    center_of_circular_point,
    Draw,
)
from bfal.specs import (
    BodySpec,
)

LABEL_YGAP_AMOUNT = 3 # how high to put the label above the box location

BOTTOM_CHIN_INX = 8
BOTTOM_NOSE_TIP_INX = 2
TOP_END_NOSE_BRIDGE_INX = 0

CHIN = 'chin'
NOSE_BRIDGE = 'nose_bridge'
NOSE_TIP = 'nose_tip'
LEFT_EYE = 'left_eye'
RIGHT_EYE = 'right_eye'
BOTTOM_LIP = 'bottom_lip'
TOP_LIP = 'top_lip'

BOTTOM_LIP_BOTTOM_INDX = 3
TOP_LIP_TOP_INDX = 3

# ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']

class FaceSpec:

    def __init__(self, location, land_marks, is_known, label, distance_value) -> None:
        self.location = location
        self.land_marks = land_marks
        self.is_known = is_known
        self.label = label
        self.distance_value = distance_value

        self.top_lip_top = land_marks.get(TOP_LIP)[TOP_LIP_TOP_INDX]
        self.bottom_lip_cottom = land_marks.get(BOTTOM_LIP)[BOTTOM_LIP_BOTTOM_INDX]
        self.mid_lip = midpoint(self.top_lip_top, self.bottom_lip_cottom)

    def drawIn(self, image, includeLandMarks=False) -> None:
        (top, right, bottom, left) = self.location

        # draw rectangle face
        cv.rectangle(
            image,
            (left, top),
            (right, bottom),
            color=(0, 255, 0),
            thickness=1
        )

        # draw landmarks
        # if includeLandMarks:
        #     for marks in self.land_marks.values():
        #         Draw.draw_point_line(
        #             image=image,
        #             points=marks,
        #             circle_color=Draw.RED,
        #             line_color=Draw.BLUE,
        #         )

        # draw label above face box
        cv.putText(
            image,
            text=f'{self.label}:({self.distance_value:.2f})',
            org=(left, top - LABEL_YGAP_AMOUNT),
            fontFace=cv.FONT_HERSHEY_PLAIN,
            color=(0, 255, 0),
            fontScale=1,
            thickness=1,
        )
        
        # draw bottom chin to bottom nose line
        # cv.line(
        #     image,
        #     pt1=self.land_marks.get(CHIN)[BOTTOM_CHIN_INX],
        #     pt2=self.land_marks.get(NOSE_BRIDGE)[TOP_END_NOSE_BRIDGE_INX],
        #     color=(0, 137, 255),
        #     thickness=2,
        # )

    def getBottomChinEyeLineDistance(self) -> float:
        left_eye_center_p = center_of_circular_point(self.land_marks.get(LEFT_EYE))
        right_eye_center_p = center_of_circular_point(self.land_marks.get(RIGHT_EYE))

        mid_eye_p = midpoint(left_eye_center_p, right_eye_center_p)

        return get_distance_of_2_points(
            pt1=mid_eye_p,
            pt2=torch.tensor(self.land_marks.get(CHIN)[BOTTOM_CHIN_INX]).to('cuda'),
        )

    def getBottomChinNoseTipDistance(self) -> float:
        return get_distance_of_2_points(
            pt1=torch.tensor(self.land_marks.get(CHIN)[BOTTOM_CHIN_INX]).to('cuda'),
            pt2=torch.tensor(self.land_marks.get(NOSE_TIP)[BOTTOM_NOSE_TIP_INX]).to('cuda'),
        )
    
    def getBottomChinMidLipDistance(self) -> float:
        return get_distance_of_2_points(
            pt1=torch.tensor(self.land_marks.get(CHIN)[BOTTOM_CHIN_INX]).to('cuda'),
            pt2=self.mid_lip,
        )
    
    @staticmethod
    def pop_fspec(faces_spec: list, bspec: BodySpec) -> 'FaceSpec':
        """
            bruteforcefuly find and remove face specs that belongs to the bspec from a lists of facespec
        """
        
        ret_val = None

        for i in range(len(faces_spec)):
            fspec = faces_spec[i]

            if bspec.owns_fspec(fspec):
                ret_val = fspec
                del faces_spec[i]
                break

        return ret_val