import cv2 as cv
import torch
import click
import serial
import pkg_resources

from bfal.scripts.core import (
    PoseYOLO,
    FaceRecognition,
    ArucoRef,
    BuiltManager,
    BFALSerialConn,
    UNKNOWN_PERSON_LABEL,
)

from bfal.specs import (
    FaceSpec,
    BodySpec,
    BuiltSpec,
)

from bfal.utils import (
    AsyncVideoCapture,
    FPS,
    MedianFilter,
    crop_9_16,
    Draw,
)

import bfal.config as cf

torch.set_default_device('cuda')

# camera defaults
TARGET_CAMERA = cf.get(cf.CAM_TARGET)
USE_LIVE_REF = cf.get(cf.CNFD_USE_LIVE_REF)
# aruco ref line config
REAL_DISTANCE = cf.get(cf.CNFD_VALUE)
UNIT = cf.get(cf.CNFD_UNIT)
PIXEL_DISTANCE = cf.get(cf.CNFD_DIST_PIXEL)
REF_LINE_Y_AXIS = cf.get(cf.CNFD_LINE_Y_AXIS)

SERIAL_RECOGNIZE_MSG = b'1'
SERIAL_NOT_RECOGNIZE_MSG = b'0'

# core
pose_yolo = PoseYOLO(pkg_resources.resource_filename('bfal', '/configs/models/yolov8-pose.pt'))
face_recg = FaceRecognition(cf.get(cf.PATH_FACES))
arc_ref = ArucoRef()
builtM = BuiltManager(cf.get(cf.PATH_BUILTS))

# load builts json
builtM.load()

# serial connection
ser = None
serial_conn = None
try:
    port = cf.get(cf.SERIAL_PORT)
    click.echo(f'Connecting to PORT: {port}')
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = cf.get(cf.SERIAL_BAUDRATE)
    ser.open()
    serial_conn = BFALSerialConn(
        serial=ser, 
        th=cf.get(cf.TH_SERIAL_CONSISTENCY_REQ),
        window=cf.get(cf.TH_SERIAL_WINDOW),
    )
    click.echo(f'Connected to port:{port}')
except Exception as e:
    click.echo(e)

# initialize video capture
cap = AsyncVideoCapture(TARGET_CAMERA)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cf.get(cf.CAM_WIDTH))
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cf.get(cf.CAM_HEIGHT))

# define FPS
fps = FPS()

# filters
arc_ref_mfilt = MedianFilter(50)
width_mfilt = MedianFilter(16)
height_mfilt = MedianFilter(16)

# 
last_person_width_read = 0
last_person_height_read = 0
last_person_label_read = None

# begin capture
cap.begin()
fps.init()
while True:
    ret, frame = cap.read()

    # init variables
    faces_count = 0
    known_faces_count = 0
    body_count = 0
    valid_body_count = 0
    person_width = 0
    person_height = 0
    person_label = None
    fdistance = None # filtered distance ref

    if not ret:
        break

    # frame = cv.flip(frame, flipCode=1)
    # frame = imutils.resize(frame, width=480)

    frame = crop_9_16(frame)
    insp_frame = frame.copy()
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # detect pose async
    pose_yolo.detect_async(rgb_frame)

    # detect face async
    faces_spec = face_recg.process(rgb_frame) # return lists of face spec

    # check for aruco distance reference
    if USE_LIVE_REF:
        arc_ref_valid = arc_ref.ref_valid(frame, filter=False)
        if arc_ref_valid:
            arc_ref_dist = arc_ref.get_distance()

            arc_ref_mfilt.insert(arc_ref_dist)
            fdistance = arc_ref_mfilt.retrieve() # get filtered distance
    else:
        fdistance = PIXEL_DISTANCE
        ArucoRef.__draw_ref_lines__(frame, REF_LINE_Y_AXIS)

    # draw face rect and landmarks
    faces_count = len(faces_spec)
    for fspec in faces_spec:
        fspec.drawIn(frame, includeLandMarks=False)
        
        if fspec.label != UNKNOWN_PERSON_LABEL:
            known_faces_count += 1

    # get lists of pose results
    pose_results = pose_yolo.get_result()

    if pose_results:
        pose_results = pose_results[0]
        body_count = len(pose_results)

        # analyze pose results check if body is aligned and match it to their corresponding faces
        # if body is aligned and face is present, check if body built and face is known
        for (keypoints, box) in zip(pose_results.keypoints.data, pose_results.boxes.data):
            body_spec = BodySpec(image=insp_frame, imageLog=frame, verbose=True, keypoints=keypoints, box=box)
            is_body_firm = body_spec.body_is_firm()
            is_head_firm = body_spec.head_is_firm()

            if is_body_firm and is_head_firm and faces_spec:
                # check if detected body is inside the reference line
                body_within_ref_line = ArucoRef.body_is_within_ref(blt_spec, arc_ref.my if USE_LIVE_REF else REF_LINE_Y_AXIS)
                if not body_within_ref_line:
                    break # ignore this detected person even has valid built

                # increment valid body count
                valid_body_count += 1

                # find the face of body_spec
                face_spec = FaceSpec.pop_fspec(faces_spec=faces_spec, bspec=body_spec)
                
                # body_spec face is present
                if face_spec != None:

                    person_label = face_spec.label
                    blt_spec = BuiltSpec(bodySpec=body_spec, faceSpec=face_spec)
                    blt_spec.drawIn(frame)

                    px_width, px_height = blt_spec.getBuilt()

                    # filter built values
                    width_mfilt.insert(px_width)
                    height_mfilt.insert(px_height)

                    fwidth = width_mfilt.retrieve()
                    fheight = height_mfilt.retrieve()

                    if fdistance:
                        # do conversion
                        rwdst_ratio = REAL_DISTANCE / fdistance

                        # real world measurement
                        rw_width = fwidth * rwdst_ratio
                        rw_height = fheight * rwdst_ratio

                        person_width = rw_width
                        person_height = rw_height
                        person_label = fspec.label              

                        last_person_width_read = rw_width
                        last_person_height_read = rw_height
                        last_person_label_read = fspec.label

                        # verify if face and builts is within the json builts
                        person_is_known = builtM.verify(person_label, (rw_width, rw_height))

                        if serial_conn:
                            # send status signal to serial port
                            message = SERIAL_RECOGNIZE_MSG if person_is_known else SERIAL_NOT_RECOGNIZE_MSG
                            serial_conn.queue(person_label, data=message)

    # log FPS
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

    # echo logs
    click.clear()
    click.echo(f'Face: Detected={faces_count}, Known={known_faces_count}')
    click.echo(f'Body: Detected={body_count}, Valid={valid_body_count}')
    click.echo('-'*64)
    click.echo(f'Result: Width={person_width:.2f}{UNIT}, Height={person_height:.2f}{UNIT}, label={person_label}')
    click.echo(f'Last Valid Result: Width={last_person_width_read:.2f}{UNIT}, Height={last_person_height_read:.2f}{UNIT}, label={last_person_label_read}')

    # draw crosshairs
    Draw.crosshairs(frame, color=Draw.ORANGE)
    # draw ruler
    # Draw.ruler(frame, org=(64, REF_LINE_Y_AXIS), gap=10)

    # show image
    cv.imshow('Built-Face Detection', frame)

    fps.update()

    key = cv.waitKey(1)
    if key == ord('q'):
        break

# cleaning
if ser:
    ser.close()
cap.release()
cv.destroyAllWindows()