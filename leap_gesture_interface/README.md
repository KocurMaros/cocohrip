# Simple-Leap-Motion-ROS2-Gesture-Interface

# ROS 2 Leap Gesture Interface

This ROS 2 package provides a basic gesture recognition node utilizing the Leap Motion controller with ROS2 environment. The node is capable of identifying specific hand gestures and provides both raw and normalized Roll, Pitch, and Yaw values of the hand.

## Features

- Recognition of predefined hand gestures using Leap Motion.
- Publishing roll, pitch, and yaw data for the detected hand.

## Dependencies

- ROS 2 (tested on Iron). Follow instructions here: https://docs.ros.org/en/iron/Installation/Ubuntu-Install-Debians.html
- Leap Motion SDK. Follow instructions here:
  https://docs.ultraleap.com/linux/?_gl=1*1otp0h2*_ga*NDE1ODUyMDIyLjE3MDAxNDc2NDE.*_ga_5G8B19JLWG*MTcwMDQ3NTAyNS40LjEuMTcwMDQ3NTAzMi41My4wLjA.

## Installation

1. **Set up a ROS 2 workspace** (skip if you already have one):

    ```bash
    mkdir -p ~/ros2_ws/src
    cd ~/ros2_ws/
    ```

2. **Clone this repository**:

    ```bash
    cd src/
    git clone https://github.com/MarekC96/leap_gesture_interface.git
    ```

3. **Install dependencies**:

    Navigate back to your ROS 2 workspace root and run:

    ```bash
    rosdep install --from-paths src --ignore-src -r -y
    ```

4. **Build the package**:

    ```bash
    colcon build --packages-select leap_gesture_interface
    ```

5. **Source the environment**:

    ```bash
    . install/setup.bash
    ```

## Usage

Run the basic gesture recognition node using the following command:

```bash
ros2 run leap_gesture_interface gesture_interface_node
```

## Recognized Gestures

- FIST - no fingers extended
- THUMB - only thumb finger extended
- POINT - only index finger extended
- PINKY - only pinky finger extended
- PISTOL - only thumb and index fingers extended
- VICTORY -only index and middle fingers extended
- HORNS - only index and ring fingers extended
- SHAKA - only thumb and pinky fingers extended
- THREE - only thumb and index and middle fingers extended
- ROCKER - only thumb and index and pinky fingers extended
- FOUR - four fingers extended
- FIVE - all fingers extended

## Published topics
The node publishes current recognized gesture of each hand on the following topics:
- leap_gesture
- leap_gesture/left
- leap_gesture/right

The node does a basic filtering based on the previous frames and publishes them on the following topics:
- leap_gesture/filtered
- leap_gesture/filtered_left
- leap_gesture/filtered_righ

The node publishes also gesture initialized gestures, this means that the gesture is published only when an initialization gesture was performed previously (for example FIST and then POINT, where FIST is initialization geture). The topics are:
- leap_gesture/gesture_initiated
- leap_gesture/gesture_initiated_left
- leap_gesture/gesture_initiated_right

The node publishes double gestures (Both hands are performing the same gesture at the same time) on topics:
- leap_gesture/double
- leap_gesture/double_filtered
- leap_gesture/double_initiated

The node publishes Roll, Pitch, Yaw of the hands either in degrees or in normalized value <-1,1> normalised from the range <-90,90> degrees. The topics are:
- leap_gesture/raw_rpy_left
- leap_gesture/raw_rpy_right
- leap_gesture/normalized_rpy_left
- leap_gesture/normalized_rpy_right



