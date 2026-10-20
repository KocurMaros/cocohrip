#ifndef MISSION_HPP
#define MISSION_HPP

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

    // Constructor
    Mission(int row, int col, const std::string& color, Task task,
            double offsetRow = 0.0, double offsetCol = 0.0);
};

#endif // MISSION_HPP
