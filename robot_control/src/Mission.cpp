#include "robot_control/Mission.hpp"

Mission::Mission(int row, int col, const std::string& color, Task task, double offsetRow, double offsetCol)
    : row(row), col(col), color(color), task(task), offsetRow(offsetRow), offsetCol(offsetCol) {}
