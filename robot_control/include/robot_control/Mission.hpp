#ifndef MISSION_HPP
#define MISSION_HPP

#include <cmath>
#include <vector>
#include <string>
#include "checkers_msgs/msg/piece.hpp" 

enum class Task {
    NONE = 0,
    ATTACH = 1,
    DETACH = 2,
};

class Mission {
public:
    int row, col;
    std::string color; 
    Task task;
    double offsetRow, offsetCol;  // piece centre off its square, in squares (game frame)

    // Chess moves (chess_msgs/ChessRobotMove) bring their own geometry;
    // unset (NaN / -1) = the checkers defaults (missionToXY, zAttach,
    // zSafeTransition, gripper_close_position)
    bool hasXY = false;
    double x = 0.0, y = 0.0;        // robot-frame fingertip target [m]
    double zGrip = std::nan("");    // fingertip height for the gripper action [m]
    double zTravel = std::nan("");  // fingertip height between squares [m]
    int gripClose = -1;             // Hand-E close target of this pick (0..255)

    // Constructor
    Mission(int row, int col, const std::string& color, Task task,
            double offsetRow = 0.0, double offsetCol = 0.0);
};

#endif // MISSION_HPP
