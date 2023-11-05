import cv2 as cv
import click
import bfal.config as cf
from bfal.utils import (
    AsyncVideoCapture,
    FPS,
    MedianFilter,
    Draw,
)
from bfal.scripts.core import ArucoRef

cam_target = cf.get(cf.CAM_TARGET)
cam_width = cf.get(cf.CAM_WIDTH)
cam_height = cf.get(cf.CAM_HEIGHT)

calibration_tolerance = cf.get(cf.CNFD_CALIB_TOL)
ref_unit = cf.get(cf.CNFD_UNIT)
distance_value = cf.get(cf.CNFD_VALUE)

acap = AsyncVideoCapture(cam_target)
acap.set(cv.CAP_PROP_FRAME_WIDTH, cam_width)
acap.set(cv.CAP_PROP_FRAME_HEIGHT, cam_height)

fps = FPS()

mfilter = MedianFilter(64)
aref = ArucoRef()

# define req variables
steady_count = 0
max_steady_count = 0
prev_distance = -1
calib_success = False

# declare escape character
UP = "\x1B[3A" # move cursor up X times
CLR = "\x1B[0K"

click.echo('\n')

# begin capturing
acap.begin()
fps.init()
while True:
    ret, frame = acap.read()

    if not ret:
        break
    
    # frame = cv.flip(frame, flipCode=1)
    # frame = crop_9_16(frame)
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # draw crosshairs
    Draw.crosshairs(image=frame, thickness=1)

    ref_valid = aref.ref_valid(frame)

    fdist = None
    if ref_valid:
        ref_dist = aref.get_distance()

        mfilter.insert(ref_dist)
        fdist = mfilter.retrieve()

        if prev_distance != fdist:
            steady_count = 0 # reset count if prev value is not the same with new one

        prev_distance = fdist
        steady_count += 1
        max_steady_count = max(max_steady_count, steady_count)
    
    click.echo(UP)
    click.echo(f'Pixel distance: {prev_distance:.4f}')
    click.echo(f'Steady Count: {steady_count}, Max: {max_steady_count}{CLR}')

    if steady_count >= calibration_tolerance:
        prev_distance = int(prev_distance)

        click.echo('-'*64)
        click.echo('Calibration is finished with ff values.')
        click.echo(f'\t{distance_value}{ref_unit} ~= {prev_distance}px')

        cf.set(cf.CNFD_DIST_PIXEL, prev_distance, override=True)
        cf.set(cf.CNFD_UNIT, ref_unit, override=True)
        cf.set(cf.CNFD_VALUE, distance_value, override=True)
        cf.set(cf.CNFD_LINE_Y_AXIS, aref.my, override=True)
        cf.save()
        calib_success = True
        break

    fps.stop()
    fps_val = fps.value()
    cv.putText(
        frame,
        text=f'FPS:{fps_val:.2f}',
        fontFace=cv.FONT_HERSHEY_PLAIN,
        org=(0, 16),
        color=(0, 255, 0),
        fontScale=1,
        thickness=2,
    )
    cv.imshow('Calibrate Distance', frame)

    fps.update()

    if cv.waitKey(1) == ord('q'):
        break

if not calib_success:
    click.echo('\tNo config changes.')

acap.release()
cv.destroyAllWindows()
