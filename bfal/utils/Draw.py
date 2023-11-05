import cv2 as cv
import bfal.config as cf
import torch
from bfal.utils import (
    get_distance_of_2_points,
)

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
PURPLE = (128,0,128)
YELLOW = (0, 255, 255)

def draw_line(
    image,
    points,
    color,
    thickness=1,
    lineType=cv.LINE_4,
):
    prev_point = None
    
    for (x, y, *_) in points:
        pnt = int(x), int(y)

        if prev_point != None:
            cv.line(
                img=image,
                pt1=prev_point,
                pt2=pnt,
                color=color,
                thickness=thickness,
                lineType=lineType,
            )

        prev_point = pnt


def draw_point_line(
    image,
    points,
    circle_color,
    line_color,
    radius=1,
    thickness=1,
):
    prev_point = None

    for (x, y, *_) in points:
        pnt = (int(x), int(y))
        cv.circle(
            image, 
            center=pnt, 
            radius=radius, 
            color=circle_color, 
            thickness=-1
        )

        if prev_point != None:
            cv.line(
                image, 
                pt1=prev_point, 
                pt2=pnt, 
                color=line_color, 
                thickness=thickness
            )

        prev_point = pnt

def draw_text(
    image,
    text,
    org,
    color,
    fontFace=cv.FONT_HERSHEY_PLAIN,
    fontScale=1,
    thickness=1,
):
    x, y = org
    org = int(x), int(y)

    cv.putText(
        img=image,
        text=text,
        org=org,
        color=color,
        fontFace=fontFace,
        fontScale=fontScale,
        thickness=thickness,
    )

def crosshairs(image, color=RED, thickness=1, lineType=cv.LINE_4):
    height, width, *_ = image.shape
    xcenter = int(width / 2)
    ycenter = int(height / 2)

    vp1 = (xcenter, 0)
    vp2 = (xcenter, height)

    hp1 = (0, ycenter)
    hp2 = (width, ycenter)

    cv.line(image, pt1=vp1, pt2=vp2, color=color, thickness=thickness, lineType=lineType)
    cv.line(image, pt1=hp1, pt2=hp2, color=color, thickness=thickness, lineType=lineType)

def ruler(image, org, line_lent=16, gap=1) -> None:
    RW_DIST = cf.get(cf.CNFD_VALUE)
    PX_DIST = cf.get(cf.CNFD_DIST_PIXEL)
    UNIT = cf.get(cf.CNFD_UNIT)

    # rw measurement to pixel ratio
    ratio = PX_DIST / RW_DIST
    # pixel gap equivalent of rw measurement gap
    px_gap = int(ratio * gap)

    ox, oy = org

    # vertical line points
    vp1 = org
    vp2 = (ox, 0) # move point to top most of image (y=0)

    # distance of vertical ruler in pixel
    ruler_px_lent = get_distance_of_2_points(torch.tensor(vp1), torch.tensor(vp2))

    # draw vertical line of ruler
    cv.line(image, pt1=vp1, pt2=vp2, color=BLUE, thickness=2)

    # draw ruler lines (samll horizontal line in ruler)
    i = 1
    for pxg in range(px_gap, int(ruler_px_lent.item()), px_gap):
        Ly = oy - pxg
        Lp1 = (ox - 3, Ly) # from original point move upward by the value of gap in px
        Lp2 = (ox + line_lent, Ly)

        cv.line(image, pt1=Lp1, pt2=Lp2, color=BLUE, thickness=1)
        cv.putText(
            image,
            text=f'{gap * i}{UNIT}',
            color=YELLOW,
            org=Lp2,
            fontFace=cv.FONT_HERSHEY_PLAIN,
            fontScale=0.8,
            thickness=1,
        )

        i += 1
