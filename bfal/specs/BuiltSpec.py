import torch
from bfal.utils import (
    get_distance_of_2_points,
    midpoint,
    Draw,
)

from bfal.specs.body_parts import (
    YOLO_LEFT_SHOULDER,
    YOLO_RIGHT_SHOULDER,
)

class BuiltSpec:

    def __init__(self, bodySpec, faceSpec) -> None:
        self.bodySpec = bodySpec
        self.faceSpec = faceSpec

        self.__init__points__()

    def __init__points__(self) -> None:
        self.top_bpoint = self.bodySpec.get_mid_top()
        self.bot_bpoint = self.bodySpec.get_mid_bottom()

        # add proportion measurements to estimate top head and bottom ground feet by the ankle
        self.top_bpoint[1] -= self.faceSpec.getBottomChinEyeLineDistance()
        self.bot_bpoint[1] += self.faceSpec.getBottomChinMidLipDistance()

    def getBuilt(self) -> float:
        # distance of estimated top and bottom point
        height = get_distance_of_2_points(
            torch.tensor(self.top_bpoint), 
            torch.tensor(self.bot_bpoint),
        )
        # distance of left and right shoulder point
        width = get_distance_of_2_points(
            self.bodySpec.get_body_point(YOLO_LEFT_SHOULDER),
            self.bodySpec.get_body_point(YOLO_RIGHT_SHOULDER),
        )

        return (width.item(), height.item())
    

    def drawIn(self, image) -> None:
        left_shoulder_p = self.bodySpec.get_body_point(YOLO_LEFT_SHOULDER)
        right_shoulder_p = self.bodySpec.get_body_point(YOLO_RIGHT_SHOULDER)
        mid_shoulder_p = midpoint(left_shoulder_p, right_shoulder_p)
        mid_shoulder_p = mid_shoulder_p

        (width, height) = self.getBuilt()

        # draw height line
        Draw.draw_line(
            image=image,
            points=[self.top_bpoint, self.bot_bpoint],
            color=Draw.RED,
            thickness=3,
        )

        # draw width line
        Draw.draw_line(
            image=image,
            points=[left_shoulder_p, right_shoulder_p],
            color=Draw.PURPLE,
            thickness=3,
        )

        # height value text
        Draw.draw_text(
            image=image,
            text=f'{height:.2f}',
            org=(self.top_bpoint[0] + 3, self.top_bpoint[1] + 10),
            color=Draw.ORANGE,
            fontScale=1,
            thickness=2,
        )

        # width value text
        Draw.draw_text(
            image=image,
            text=f'{width:.2f}',
            org=(mid_shoulder_p[0] + 2, mid_shoulder_p[1] - 4),
            color=Draw.ORANGE,
            fontScale=1,
            thickness=2,
        )