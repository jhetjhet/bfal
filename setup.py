from setuptools import setup, find_packages

setup(
    name='bfal',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click>=8.1.7',
        'opencv-python>=4.8.0.76',
        'imutils>=0.5.4',
        'dlib>=19.24.99',

        'tensorflow>=2.13.0',
        'torch>=2.0.1',
        'torchaudio>=2.0.2',
        'torchvision>=0.15.2',

        'face-recognition>=1.3.0',
        'ultralytics>=8.0.184',
    ],
    package_data={
        'configs': ['config.ini', 'yolov8-pose.pt'],
    },
    entry_points={
        'console_scripts': [
            'bfal = bfal.scripts.bfal:main',
        ],
    },
)