# BFAL
## _Built and Face Image Recognition_

It produces fairly estimation of person's width and height with face recognition.

Prerequisites:
- NVIDIA graphics card
- system that support CUDA and CUDNN
- [dlib] build with CUDA enabled.
- [tensorflow] with CUDA enabled.
- [pytorch] with CUDA enabled.

To reduce potential problems, consider applying the following methods, which were used during the development of this project:
- Use linux environment.
- to build dlib with CUDA and CUDNN supported follow the guide on this [post]
- use conda environment.

## Installation
It is important that you already have the required prerequisite before installing this program.
```sh
git clone https://github.com/jhetjhet/bfal.git
cd bfal
python setup.py install
```

## Usage
The program has two commands `calibrate` and `detect`.
```sh
bfal [OPTIONS] COMMAND [ARGS]...
```
##### Command line arguments

| Option | Description |
| ------ | ------ |
| `--help` | Display this help message. |
| `-c` `--camera` | Target camera.  [default: 0] |
| `-r` `--res` | Camera Resolution  [default: 1080, 720] |
| `-o` `--override` | Save all overriden properties to config file. |
#### Calibration
For this mode, you'll need a printed Aruco image as a reference. Position it in a straight line, vertically aligned with the camera, and measure the distance to the mid-point of the image, as demonstrated below. The goal of this calibration is to store the reference distance in the configuration file for later use in the detection process.

Aruco image file:\
[aruco_4x4_200_0.pdf](https://github.com/jhetjhet/bfal/files/13259270/aruco_4x4_200_0.pdf)\
[aruco_4x4_200_1.pdf](https://github.com/jhetjhet/bfal/files/13259271/aruco_4x4_200_1.pdf)


Alignment of two aruco image reference:\
![Screenshot from 2023-11-05 15-39-56](https://github.com/jhetjhet/bfal/assets/74247535/0810d241-d364-4d74-8b14-db065dafb233)


> Note: The result of this calibration will be saved on `config.ini` file.
##### Calibration command line arguments

| Option | Description |
| ------ | ------ |
| `--help` | Display this help message. |
| `-t` `--tol` | Required number of steady value.  [default: 128] |
| `-d` `--dist` | Value of real distance.  [default: 90.5] |
| `-u` `--unit` | Unit of measurement used for real distance.  [default: cm |

#### Detection
Execute the core functionality of this program.

> Note: You dont have to provide all this options each time you run the program as it is save in `config.ini` file.
> Note: To save the provided options in `config.ini` make sure to use `-o` `--override` option.

##### Detection command line arguments
| Option | Description |
| ------ | ------ |
| `--help` | Display this help message. |
| `-p` `--port` | serial connection port address |
| `-b` `--baudrate` | serial connection baudrate  [default: 9600] |
| `-lr` `--live-aref` | use live aruco distance reference. |
| `-fp` `--faces-path` | Directory of known faces. |
| `-bp` `--builts-path` | File location of persons builts. |
| `-fv` `--face-visibility` | Minimum visibility required for each face point  [default: 0.5] |
| `-bv` `--body-visibility` | Minimum average visibility required for body points  [default: 0.9] |
| `-al` `--ankle-line` | Maximum tolerated distance between two ankle points based on their mid Y-axis  [default: 6] |
| `-sl` `--shoulder-line` | Maximum tolerated distance between two shoulder points based on their mid Y-axis [default: 6.0] |
| `-ha` `--head-angle` | Allowed angle range for the head, measured from the nose to the mid-eye point [default: 7.0] |
| `-kb` `--knee-bend` | Tolerance for knee bend, calculated by comparing the distance of two endpoints to the sum of distances between each point [default: 0.5] |
| `-bt` `--built-tolerance` | Maximum tolerance for unit value based on the REFERENCE section  [default: 3] |
| `-arl` `--aruco-line` | Maximum tolerated distance between two Aruco midpoints based on their mid Y-axis [default: 2] |
| `-abl` `--aruco-body-line` | Threshold for the maximum allowed distance between the bottom body point and the Aruco line reference  [default: 6] |
| `-sc` `--serial-consistency` | Number of constant messages required before sending a serial message  [default: 10] |
| `-sw` `--serial-window` | Maximum time duration for a message to be considered valid as part of the constant message  [default: 1000] |

## Default Configs
Located on `./bfal/configs/config.ini`
```sh
[CAMERA]
width = 1080
height = 720
target = 0

[REFERENCE]
calibration_tolerance = 128
use_live_ref = False
distance_unit = 'cm'
distance_value = 90.5
distance_pixel = 321
aruco_line_y_axis = 658

[THRESHOLDS]
face_visibility = 0.5
body_visibility = 0.9
ankle_line = 6
shoulder_line = 6.0
head_angle = 7.0
knee_bend = 0.5
built_tolerance = 3
aruco_line = 2
aruco_body_line = 6
serial_consistency_req = 10
serial_window = 1000

[PATHS]
known_faces = ''
builts_json = ''

[SERIAL_CONN]
port = ''
baudrate = 9600
```

## Serial Communication
If the provided port is valid, the program will send a `1` over the serial communication if both the person's body and face are recognized. Otherwise, it will send `0`.


[//]:()
   [dlib]: <https://github.com/davisking/dlib>
   [tensorflow]: <https://www.tensorflow.org/install/pip>
   [pytorch]: <https://pytorch.org/get-started/locally/>
   [post]: <https://gist.github.com/nguyenhoan1988/ed92d58054b985a1b45a521fcf8fa781>
 
