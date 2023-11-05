import cv2 as cv
import torch

from bfal.utils import (
    points_aligned_by_axis,
    normalize_vector,
    curveness_difference,
    extend_line_to_y,
    midpoint,
    Draw,
)

from bfal.specs.body_parts import *
from bfal.specs import FaceSpec

import bfal.config as cf

TOTAL_BODY_VISIBILITY = cf.get(cf.TH_BODY_VISIBILITY)
ANKLE_LINE_TH = cf.get(cf.TH_ANKLE_LINE)
SHOULDER_LINE_TH = cf.get(cf.TH_SHOULDER_LINE)
FACE_VISIBILITY = cf.get(cf.TH_FACE_VISIBILITY)
HEAD_ANGLE = cf.get(cf.TH_HEAD_ANGLE)
KNEE_BEND = cf.get(cf.TH_KNEE_BEND)

class BodySpec:

    def __init__(self, image, keypoints, box, imageLog=None, verbose=False) -> None:
        self.image = image
        self.keypoints = keypoints
        self.box = box
        self.imageLog = imageLog
        self.verbose = verbose

    def get_body_point(self, part: int, incV=False):

        if not incV:
            return self.keypoints[part][:2]

        return self.keypoints[part]

    def all_keypoints_visibility_mean(self):
        return torch.mean(self.keypoints, dim=0)[2]
    
    def feet_aligned_to_hips(self):
        """
        """
        left_hip = self.get_body_point(YOLO_LEFT_HIP)
        right_hip = self.get_body_point(YOLO_RIGHT_HIP)

        left_straightness = curveness_difference([left_hip, self.get_body_point(YOLO_LEFT_KNEE), self.get_body_point(YOLO_LEFT_ANKLE)])
        right_straightness = curveness_difference([right_hip, self.get_body_point(YOLO_RIGHT_KNEE), self.get_body_point(YOLO_RIGHT_ANKLE)])

        is_left_aligned = left_straightness <= KNEE_BEND
        is_right_aligned = right_straightness <= KNEE_BEND

        # image draw logging
        if self.verbose:
            left_feet_color = Draw.GREEN if is_left_aligned else Draw.RED
            right_feet_color = Draw.GREEN if is_right_aligned else Draw.RED

            Draw.draw_point_line(
                image=self.imageLog,
                points=[left_hip, self.get_body_point(YOLO_LEFT_KNEE), self.get_body_point(YOLO_LEFT_ANKLE)],
                circle_color=left_feet_color,
                line_color=left_feet_color,
                radius=2,
                thickness=2,
            )

            Draw.draw_point_line(
                image=self.imageLog,
                points=[right_hip, self.get_body_point(YOLO_RIGHT_KNEE), self.get_body_point(YOLO_RIGHT_ANKLE)],
                circle_color=right_feet_color,
                line_color=right_feet_color,
                radius=2,
                thickness=2,
            )

            Draw.draw_text(
                image=self.imageLog,
                text=f'{left_straightness:.2f}',
                org=self.get_body_point(YOLO_LEFT_KNEE),
                color=Draw.ORANGE,
                thickness=2,
            )

            Draw.draw_text(
                image=self.imageLog,
                text=f'{left_straightness:.2f}',
                org=self.get_body_point(YOLO_RIGHT_KNEE),
                color=Draw.ORANGE,
                thickness=2,
            )

        return is_left_aligned and is_right_aligned
    
    def ankle_is_aligned(self):
        left_ankle_p = self.get_body_point(YOLO_LEFT_ANKLE)
        right_ankle_p = self.get_body_point(YOLO_RIGHT_ANKLE)

        left_shoulder_x = self.get_body_point(YOLO_LEFT_SHOULDER)[0]
        right_shoulder_x = self.get_body_point(YOLO_RIGHT_SHOULDER)[0]

        # check if both ankles x's is within the shoulders x's
        left_ankle_within_shoulder_x = left_ankle_p[0] <= left_shoulder_x
        right_ankle_within_shoulder_x = right_ankle_p[0] >= right_shoulder_x

        # check if ankles x's crosses
        ankles_crossed = right_ankle_p[0] > left_ankle_p[0]

        mid_angkle_p_y = int((left_ankle_p[1] + right_ankle_p[1]) / 2)

        is_aligned = points_aligned_by_axis([left_ankle_p, right_ankle_p], mid_angkle_p_y, th=ANKLE_LINE_TH)
        is_aligned = is_aligned and left_ankle_within_shoulder_x and right_ankle_within_shoulder_x and not ankles_crossed

        # draw ankle alignment log
        if self.verbose:
            line_left = self.box[0] # start x
            line_right = self.box[2] # end x

            Draw.draw_line(
                image=self.imageLog,
                points=[
                    (line_left, mid_angkle_p_y + ANKLE_LINE_TH),
                    (line_right, mid_angkle_p_y + ANKLE_LINE_TH),
                ],
                color=Draw.BLUE,
                thickness=1,
            )
            Draw.draw_line(
                image=self.imageLog,
                points=[
                    (line_left, mid_angkle_p_y - ANKLE_LINE_TH),
                    (line_right, mid_angkle_p_y - ANKLE_LINE_TH),
                ],
                color=Draw.BLUE,
                thickness=1,
            )

            ankle_color = Draw.GREEN if is_aligned else Draw.RED
            Draw.draw_point_line(
                image=self.imageLog, 
                points=[left_ankle_p, right_ankle_p], 
                line_color=ankle_color,
                circle_color=ankle_color, 
                radius=2, 
                thickness=2
            )

        return is_aligned
    
    def shoulder_is_aligned(self):
        left_shoulder_p = self.get_body_point(YOLO_LEFT_SHOULDER)
        right_shoulder_p = self.get_body_point(YOLO_RIGHT_SHOULDER)

        mid_angkle_p_y = int((left_shoulder_p[1] + right_shoulder_p[1]) / 2)

        is_aligned = points_aligned_by_axis([left_shoulder_p, right_shoulder_p], mid_angkle_p_y, th=SHOULDER_LINE_TH)

        # draw logs
        if self.verbose:
            # draw shoulder alignment
            line_left = int(self.box[0])
            line_right = int(self.box[2])
            
            Draw.draw_line(
                image=self.imageLog,
                points=[
                    (line_left, mid_angkle_p_y + SHOULDER_LINE_TH),
                    (line_right, mid_angkle_p_y + SHOULDER_LINE_TH),
                    (line_left, mid_angkle_p_y - SHOULDER_LINE_TH),
                    (line_right, mid_angkle_p_y - SHOULDER_LINE_TH),
                ],
                color=Draw.BLUE,
                thickness=1,
            )

            # draw shoulder line and point
            shoulder_draw_color = Draw.GREEN if is_aligned else Draw.RED
            Draw.draw_point_line(
                image=self.imageLog, 
                points=[left_shoulder_p, right_shoulder_p], 
                line_color=shoulder_draw_color,
                circle_color=shoulder_draw_color, 
                radius=2, 
                thickness=2
            )

        return is_aligned

    def body_is_firm(self) -> bool:
        """
            Examine if detected body from result given is standing.

            Conditions:
                1. 90% of body keypoints is visible
                2. lef and right feet is in same x axis
                3. left and right shoulder point is in same y axis
                4. left and right ankle point is in same y axis
        """

        # condition 1
        body_visibility = self.all_keypoints_visibility_mean()
        is_visible = body_visibility >= TOTAL_BODY_VISIBILITY

        # draw visibility log
        if self.verbose:
            visibility_color = Draw.GREEN if is_visible else Draw.RED
            # draw box
            (x, y, w, h, *_) = self.box
            cv.rectangle(
                self.imageLog,
                pt1=(int(x), int(y)),
                pt2=(int(w), int(h)),
                color=visibility_color,
                thickness=2,
            )

        if not is_visible:
            return False

        shoulder_aligned = self.shoulder_is_aligned()
        # print(f'Shoulder aligned: {shoulder_aligned}')



        feet_alined = self.feet_aligned_to_hips()
        # print(f'Feet aligned: {feet_alined}')

        ankle_alinged = self.ankle_is_aligned()
        # print(f'Ankle aligned: {ankle_alinged}')


        return shoulder_aligned and feet_alined and ankle_alinged

    def head_in_right_angle(self):
        nose_p = self.get_body_point(YOLO_NOSE)
        left_eye_p = self.get_body_point(YOLO_LEFT_EYE)
        right_eye_p = self.get_body_point(YOLO_RIGHT_EYE)

        mid_eye_p = ((left_eye_p[0] + right_eye_p[0]) / 2, (left_eye_p[1] + right_eye_p[1]) / 2)
        shoulder_mid_y = (self.get_body_point(YOLO_LEFT_SHOULDER)[1] + self.get_body_point(YOLO_RIGHT_SHOULDER)[1]) / 2
        # ears_mid_y = (self.keypoints[3][1] + self.keypoints[4][1]) / 2

        # get nose to mid eye point vector
        nose_meye_v = (nose_p[0] - mid_eye_p[0], nose_p[1] - mid_eye_p[1])
        nose_meye_v = normalize_vector(nose_meye_v)

        # nose head angle from mid eye point
        head_line_angle = torch.rad2deg(torch.atan2(*nose_meye_v))
        head_line_is_aligned = torch.abs(head_line_angle) <= HEAD_ANGLE

        # check if nose y is above ears mid y
        # nose_below_ears = ears_mid_y < nose_p[1]

        # draw log
        if self.verbose:
            head_draw_color =  Draw.GREEN if head_line_is_aligned else Draw.RED

            # draw keypoints indications
            Draw.draw_point_line(
                image=self.imageLog, 
                points=[
                    self.get_body_point(YOLO_RIGHT_EARS),
                    self.get_body_point(YOLO_RIGHT_EYE),
                    self.get_body_point(YOLO_NOSE),
                    self.get_body_point(YOLO_LEFT_EYE),
                    self.get_body_point(YOLO_LEFT_EARS),
                ], 
                line_color=head_draw_color, 
                circle_color=head_draw_color,
                radius=2, 
                thickness=2
            )

            # point where shoulder mid y and mid eye and nose point intersect
            head_line_mid_shoulder_p = (int(extend_line_to_y(
                mid_eye_p[0], # x1
                mid_eye_p[1], # y1
                nose_p[0], # x2
                nose_p[1], # y2
                shoulder_mid_y,
            )), int(shoulder_mid_y))

            lent = 32
            mid_eye_p_extended = (int(mid_eye_p[0] - (nose_meye_v[0] * lent)), int(mid_eye_p[1] - (nose_meye_v[1] * lent)))

            Draw.draw_line(
                image=self.imageLog,
                points=[mid_eye_p_extended, head_line_mid_shoulder_p],
                color=head_draw_color,
                thickness=2,
            )

            Draw.draw_point_line(
                image=self.imageLog,
                points=[head_line_mid_shoulder_p],
                circle_color=head_draw_color,
                line_color=head_draw_color,
                radius=2,
            )

            Draw.draw_text(
                image=self.imageLog,
                text=f"{head_line_angle:.2f}",
                org=nose_p,
                color=Draw.ORANGE
            )

        return head_line_is_aligned

    def head_is_firm(self):
        # check face keypoints visibility
        all_face_keypoints_visible = True

        for (*_, v) in self.keypoints[:4]: # face points only 0-4
            if v <= FACE_VISIBILITY:
                all_face_keypoints_visible = False
                break

        # draw face visibility log
        if self.verbose:
            face_stat_message = "STEADY" if all_face_keypoints_visible else "FACE NOT SHOWN ALL"
            face_stat_color = Draw.GREEN if all_face_keypoints_visible else Draw.GREEN

            Draw.draw_text(
                image=self.imageLog,
                text=face_stat_message,
                org=self.get_body_point(YOLO_RIGHT_SHOULDER),
                color=face_stat_color,
            )

        if not all_face_keypoints_visible:
            return False
        

        head_straight = self.head_in_right_angle()

        return head_straight
    

    def get_mid_point(self):
        left_hip = self.get_body_point(YOLO_LEFT_HIP)
        right_hip = self.get_body_point(YOLO_RIGHT_HIP)

        left_shoulder = self.get_body_point(YOLO_LEFT_SHOULDER)
        right_shoulder = self.get_body_point(YOLO_RIGHT_SHOULDER)

        mid_hip_p = midpoint(left_hip, right_hip)
        mid_shoulder_p = midpoint(left_shoulder, right_shoulder)

        mid_mh_ms_p = midpoint(mid_hip_p, mid_shoulder_p)

        return mid_mh_ms_p

    def get_mid_top(self):
        left_eye = self.get_body_point(YOLO_LEFT_EYE)
        right_eye = self.get_body_point(YOLO_RIGHT_EYE)

        mid_eye_p = midpoint(left_eye, right_eye)

        return [self.get_mid_point()[0], mid_eye_p[1]]
    
    def get_mid_bottom(self):
        left_ankle = self.get_body_point(YOLO_LEFT_ANKLE)
        right_ankle = self.get_body_point(YOLO_RIGHT_ANKLE)

        mid_ankle_p = midpoint(left_ankle, right_ankle)

        return [self.get_mid_point()[0], mid_ankle_p[1]]
    

    def owns_fspec(self, fspec: FaceSpec) -> bool:
        """
            > the location here assumes that is came for face_recognition location
            with ff format (y, w, h, x)

            > it will check if location boundery is belong to this body by
            checking wehether the nose point is inside this boundary
        """
        (y1, x2, y2, x1) = fspec.location
        (nx, ny) = self.get_body_point(YOLO_NOSE)
        
        return x1 <= nx <= x2 and y1 <= ny <= y2