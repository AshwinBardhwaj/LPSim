#pragma once

struct B18TrajectorySample {
  int vehicleId;
  unsigned int pathInit, pathCurr;
  float position, speed;
  unsigned short active, lane;
};
