import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'chess_game'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    package_data={package_name + '.vision': ['*.npz', '*.onnx']},
    install_requires=['setuptools', 'chess'],
    zip_safe=True,
    maintainer='Eduard Zelenay',
    maintainer_email='eduard.zelenay@airvolute.com',
    description='Chess against the UR5e: piece recognition, rules, swappable engines',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': ['chess_node = chess_game.chess_node:main']},
)
