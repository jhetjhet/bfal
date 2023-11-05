import face_recognition
import numpy as np
import imutils
import click
from threading import Thread
from imutils import paths
from os import (
    listdir,
    path,
)
from bfal.utils import (
    find_intersection,
    get_distance_of_2_points,
    is_value_within,
    Draw,
)

from bfal.specs import (
    FaceSpec,
    BuiltSpec,
)

import bfal.config as cf

UNKNOWN_PERSON_LABEL = "unknown"
FACES_IMAGE_READING_WIDTH = 420

class FaceRecognition:

    def __init__(self, known_faces_path: str) -> None:
        self.root_path = known_faces_path
        self.__load_known_faces_encodings__()

    def __load_known_faces_encodings__(self) -> None:
        self.known_faces_encodings = []
        self.labels = []
        click.echo('Reading all known faces...')
        for folder in listdir(self.root_path):
            click.echo(f'Face for {folder}:')
            folder_path = path.join(self.root_path, folder)
            if path.isdir(folder_path):
                for img_path in paths.list_images(folder_path):
                    image = face_recognition.load_image_file(img_path)
                    # resize if width is greater than min width required
                    if image.shape[1] > FACES_IMAGE_READING_WIDTH:
                        image = imutils.resize(image=image, width=FACES_IMAGE_READING_WIDTH)
                    img_encoding = face_recognition.face_encodings(image)[0] # single face only
                    self.known_faces_encodings.append(img_encoding)
                    self.labels.append(folder)
                    click.echo(f'\t{img_path}')
    
    def __process_encodings__(self, rgb_image, face_locations) -> None:
        self.face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    def __process_landmarks__(self, rgb_image, face_locations) -> None:
        self.faces_landmarks = face_recognition.face_landmarks(rgb_image, face_locations)

    def process(self, rgb_image) -> [FaceSpec]:
        face_locations = face_recognition.face_locations(rgb_image, model='cnn')

        t1 = Thread(target=self.__process_encodings__, args=(rgb_image, face_locations, ))
        t2 = Thread(target=self.__process_landmarks__, args=(rgb_image, face_locations, ))

        t1.start()
        t2.start()
        
        t1.join()
        t2.join()

        faces_spec = []

        for (face_encoding, land_marks, location) in zip(self.face_encodings, self.faces_landmarks, face_locations):
            matches = face_recognition.compare_faces(self.known_faces_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_faces_encodings, face_encoding)
            nlabel = UNKNOWN_PERSON_LABEL
            min_indx = np.argmin(face_distances)

            is_known = matches[min_indx]

            if is_known:
                nlabel = self.labels[min_indx]

            fspec = FaceSpec(
                location=location,
                land_marks=land_marks,
                is_known=is_known,
                label=nlabel,
                distance_value=face_distances[min_indx],
            )

            faces_spec.append(fspec)

        return faces_spec


from threading import (
    Event,
    Thread,
)
from ultralytics import YOLO

class PoseYOLO:

    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)
        
        self.__detect_evet__ = Event()
        self.__detect_thread__ = None

        self.__detect_evet__.set()
    
    def detect(self, image) -> None:
        self.results = self.model(image, verbose=False)
        if not self.__detect_evet__.is_set():
            self.__detect_evet__.set()
 
    def detect_async(self, image) -> None:
        self.__detect_evet__.clear()
        self.__detect_thread__ = Thread(target=self.detect, args=(image,))
        self.__detect_thread__.start()

    def get_result(self):
        if not self.__detect_evet__.is_set():
            self.__detect_evet__.wait()
        return self.results
    


import cv2 as cv
import torch

"""
    This will run the detection of aruco (2 aruco) that will serve as a reference distance
    on real world measurement.


"""
ARUCO_REFS_ID = (0, 1)
arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)
arucoParams = cv.aruco.DetectorParameters()

class ArucoRef:

    def __init__(self) -> None:
        
        self.detector = cv.aruco.ArucoDetector(dictionary=arucoDict, detectorParams=arucoParams)

        self.arc0 = None
        self.arc1 = None
        self.valid = False

    def find_aruco_ref(self, image, verbose=True) -> bool:
        corners, markerIDs, _ = self.detector.detectMarkers(image)
        self.corners = corners
        self.markerIDs = markerIDs

        self.arc0 = None
        self.arc1 = None
        self.valid = False

        if verbose:
            cv.aruco.drawDetectedMarkers(image, corners)

        if len(corners) > 0:
            
            if np.in1d(ARUCO_REFS_ID, markerIDs).all():
                self.valid = True

            for (corner, id) in zip(corners, markerIDs):
                corner = corner[0]
                id = id[0]

                p0, p1, p2, p3 = corner

                midp = find_intersection(p0, p2, p1, p3)
                midp = np.int_(midp)
                
                if id == 0:
                    self.arc0 = midp
                elif id == 1:
                    self.arc1 = midp

                if verbose:
                    cv.circle(
                        image,
                        center=midp,
                        radius=8,
                        color=Draw.PURPLE,
                        thickness=-1,
                    )

                    # for i in range(len(corner)):
                    #     p = np.int_(corner[i])
                    #     cv.circle(
                    #         image,
                    #         center=p[:2],
                    #         radius=16,
                    #         color=(255, 0, 0),
                    #         thickness=-1,
                    #     )

                    #     cv.putText(
                    #         image,
                    #         text=f'{i}',
                    #         org=p[:2],
                    #         fontFace=cv.FONT_HERSHEY_PLAIN,
                    #         color=(0, 0, 255),
                    #         fontScale=4,
                    #         thickness=3
                    #     )
            
            if self.valid and verbose:
                Draw.draw_line(
                    image=image,
                    points=[self.arc0, self.arc1],
                    color=Draw.GREEN,
                    thickness=1,
                )

        return self.valid


    def is_aligned(self, image, verbose=True) -> bool:
        # check if two aruco is within the same y axis
        if not self.valid:
            return False        

        *_, y1 = self.arc0
        *_, y2 = self.arc1

        self.my = int((y1 + y2) / 2)
        th = cf.get(cf.TH_ARUCO_LINE)

        aligned = is_value_within(y1, self.my, th) and is_value_within(y2, self.my, th)

        if verbose:
            ArucoRef.__draw_ref_lines__(image, self.my, aligned)

        return aligned

    @staticmethod
    def __draw_ref_lines__(image, line_y_axis, isAligned=True):
        _, width = image.shape[:2]
        body_line_th = cf.get(cf.TH_ARUCO_BODY_LINE)
        th = cf.get(cf.TH_ARUCO_LINE)

        # draw boundary line for aruco ref alignment
        line_bound_1 = (0, line_y_axis + th), (width, line_y_axis + th)
        line_bound_2 = (0, line_y_axis - th), (width, line_y_axis - th)

        bound_color = Draw.BLUE if isAligned else Draw.RED
        Draw.draw_line(
            image=image,
            points=line_bound_1,
            color=bound_color,
            thickness=1,
        )
        Draw.draw_line(
            image=image,
            points=line_bound_2,
            color=bound_color,
            thickness=1,
        )

        # draw ref line body boundary
        body_bound_line1 = (0, line_y_axis + body_line_th), (width, line_y_axis + body_line_th)
        body_bound_line2 = (0, line_y_axis - body_line_th), (width, line_y_axis - body_line_th)

        Draw.draw_line(
            image=image,
            points=body_bound_line1,
            color=Draw.YELLOW,
            thickness=1,
            lineType=cv.LINE_AA,
        )
        Draw.draw_line(
            image=image,
            points=body_bound_line2,
            color=Draw.YELLOW,
            thickness=1,
            lineType=cv.LINE_AA,
        )

    def ref_valid(self, image, verbose=True) -> bool:
        arc_present = self.find_aruco_ref(image, verbose)
        if not arc_present:
            return False
        
        is_aligned = self.is_aligned(image, verbose)
        
        return is_aligned

    def get_distance(self) -> float:
        return get_distance_of_2_points(torch.from_numpy(self.arc0), torch.from_numpy(self.arc1))
    
    @staticmethod
    def body_is_within_ref(builtSpec: BuiltSpec, my) -> bool:
        # check if bottom point of body is aligned to the aruco reference line
        body_bottom = builtSpec.bot_bpoint

        return is_value_within(body_bottom[1], my, cf.get(cf.TH_ARUCO_BODY_LINE))

import json

class BuiltManager:
    """
        Loads the json file that contains all body width and height along with the owners name.

        with ff format:
            [
                {
                    label: "string label",
                    width: number,
                    height: number,
                },
                .,
                .,
                .,
            ]
    """

    def __init__(self, builts_path) -> None:
        self.builts_path = builts_path
        self.builts_indx = {}

    def __indexized__(self) -> None:
        """
            re arranged builts (array of objects) to be accessible easily
            searchable via label by mapping its index into another object
            containing label along its index position within the builts
            array of objects
        """

        builts_indx = {} # label index map

        for i in range(len(self.builts)):
            built = self.builts[i]
            builts_indx[built.get('label')] = i

        self.builts_indx = builts_indx

    def load(self) -> None:
        
        with open(self.builts_path, 'r') as builts_file:
            self.builts = json.load(builts_file)

        self.__indexized__()

    def verify(self, label, built) -> bool:
        bindx = self.builts_indx.get(label)

        if bindx == None:
            return False

        rbuilt = self.builts[bindx]
        rwidth = rbuilt.get('width')
        rheight = rbuilt.get('height')

        if rwidth and rwidth:
            tolerance = cf.get(cf.TH_BUILT_TOLERANCE)
            width, height = built
            # disregard floating value
            width = int(width)
            height = int(height)
            
            return is_value_within(width, rwidth, tolerance) and is_value_within(height, rheight, tolerance)

        return False
    
import serial
import time

class BFALSerialConn:
    """
        handle recognition results and send to serial port without flooding it.

        to not flood the serial connection ff is considered:
            - before sending success signal to serial comm, success result must meet required number of count
            - each queue added has a life (window) which is defined in integer milliseconds
    """

    def __init__(self, serial: serial.Serial, th:int, window=500, end=b'\n') -> None:
        self.__TIME = 'time'
        self.__COUNT = 'count'
        
        self.serial = serial
        self.th = th
        self.window = window
        self.end = end
        self.valid_count = 0
        """
            queues storage contains ff structure
            {
                "label": { time: date_time_ms, count: integer },
                "label": { time: date_time_ms, count: integer },
                .
                .
                .
            }
        """
        self._queues = {}

    def queue(self, label: str, data) -> None:
        if self._queues.get(label) == None:
            # queue new label with initial count
            self._queues[label] = {
                self.__TIME: None,
                self.__COUNT: 0,
            }

        queue = self._queues[label]
        
        if queue[self.__TIME] and (self.__get_ms_time__() - queue[self.__TIME]) >= self.window:
            queue[self.__COUNT] = 0 # reset

        # increase success count
        queue[self.__COUNT] += 1
        # set last inc time
        queue[self.__TIME] = self.__get_ms_time__()

        # check if valid_count exceeds threshhold
        if queue[self.__COUNT] >= self.th:
            # send success signal to serial comm
            self.__send_data__(data=data)

            # remove to queues
            del self._queues[label]

    def __get_ms_time__(self) -> int:
        return int(time.time() * 1000)
    
    def __send_data__(self, data) -> None:
        self.serial.write(data)
        self.serial.write(self.end)