import configparser
import ast
import sys
import pkg_resources

INI_CONFIG_PATH = pkg_resources.resource_filename('bfal', '/configs/config.ini')

config = configparser.ConfigParser()

# config storage to allow override value during runtime only
__config_data = {}
# map key to sections of config ini
__config_section_map = {}

def __load_config_data():
    ret = config.read(INI_CONFIG_PATH)
    if not ret:
        sys.exit(f'Error reading config file from {INI_CONFIG_PATH}.') 
    for section in config.sections():
        for key in config[section].keys():
            __config_section_map[key] = section
            __config_data[key] = ast.literal_eval(config[__config_section_map[key]][key])

def get(key: str):
    return __config_data[key]

def set(key: str, value, override=False):
    __config_data[key] = value # runtime override
    if override: # directly put value on config this will override the ini file when saved
        data = str(value) if type(value) != str else f"'{value}'"
        config[__config_section_map[key]][key] = data

def save():
    with open(INI_CONFIG_PATH, 'w') as ini_file:
        config.write(ini_file)

print(f'Reading config file from {INI_CONFIG_PATH}')
__load_config_data()

# CONFIG KEYS
# CAMERA
CAM_WIDTH = 'width'
CAM_HEIGHT = 'height'
CAM_TARGET = 'target'
# REFERENCE
CNFD_CALIB_TOL = 'calibration_tolerance'
CNFD_USE_LIVE_REF = 'use_live_ref'
CNFD_UNIT = 'distance_unit'
CNFD_VALUE = 'distance_value'
CNFD_DIST_PIXEL = 'distance_pixel'
CNFD_LINE_Y_AXIS = 'aruco_line_y_axis'
# THRESHOLDS
TH_BODY_VISIBILITY = 'body_visibility'
TH_FACE_VISIBILITY = 'face_visibility'
TH_ANKLE_LINE = 'ankle_line'
TH_SHOULDER_LINE = 'shoulder_line'
TH_HEAD_ANGLE = 'head_angle'
TH_KNEE_BEND = 'knee_bend'
TH_BUILT_TOLERANCE = 'built_tolerance'
TH_ARUCO_LINE = 'aruco_line'
TH_ARUCO_BODY_LINE = 'aruco_body_line'
TH_SERIAL_CONSISTENCY_REQ = 'serial_consistency_req'
TH_SERIAL_WINDOW = 'serial_window'
# PATHS
PATH_FACES = 'known_faces'
PATH_BUILTS = 'builts_json'
# SERIAL CONNECTION
SERIAL_PORT = 'port'
SERIAL_BAUDRATE = 'baudrate'