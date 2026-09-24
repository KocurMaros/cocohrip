from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot_control_amavet'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='amavet',
    maintainer_email='amavet@todo.todo',
    description='Robot control package for UR5e with MoveIt - Amavet version',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'move_ur5e_node = robot_control_amavet.move_ur5e_node:main',
        ],
    },
)
