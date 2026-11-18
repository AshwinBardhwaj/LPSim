#pragma once
#include "lpsim/vehicle.h"
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Applied before any host-to-device upload. Unlisted vehicles keep legacy defaults.
inline void applyVehicleProfiles(std::vector<LC::B18TrafficVehicle>& vehicles) {
  const char* filename = std::getenv("LPSIM_VEHICLE_PROFILES");
  if (!filename || !*filename) return;
  std::ifstream input(filename);
  std::string line;
  if (!input || !std::getline(input, line)) throw std::runtime_error("Cannot read vehicle profiles");
  if (!line.empty() && line.back() == '\r') line.pop_back();
  if (line != "vehicle_id,type_id,min_gap_m,time_headway_s,max_speed_mps,accel_mps2,decel_mps2")
    throw std::runtime_error("Invalid vehicle profile header");
  std::unordered_map<int, LC::B18TrafficVehicle*> byId;
  for (auto& v : vehicles) byId.emplace(v.id, &v);
  std::unordered_set<int> seen;
  struct Profile { int id, type; float gap, headway, speed, accel, decel; };
  std::vector<Profile> profiles;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (std::count(line.begin(), line.end(), ',') != 6 || line.front() == ',' || line.back() == ',' || line.find(",,") != std::string::npos)
      throw std::runtime_error("Expected seven nonempty vehicle profile fields");
    for (char& c : line) if (c == ',') c = ' ';
    Profile p; std::istringstream row(line); std::string extra;
    if (!(row >> p.id >> p.type >> p.gap >> p.headway >> p.speed >> p.accel >> p.decel) || (row >> extra)
        || p.type < 0 || p.type > 65535 || !byId.count(p.id) || !seen.insert(p.id).second
        || !std::isfinite(p.gap) || !std::isfinite(p.headway) || !std::isfinite(p.speed)
        || !std::isfinite(p.accel) || !std::isfinite(p.decel)
        || p.gap < 0 || p.gap > 65535 || p.headway <= 0 || p.speed <= 0 || p.accel <= 0 || p.decel <= 0)
      throw std::runtime_error("Invalid/duplicate/unknown vehicle profile: " + line);
    profiles.push_back(p);
  }
  for (const auto& p : profiles) {
    auto& v = *byId.at(p.id);
    v.vehicleType = p.type; v.minimumGapM = p.gap; v.desiredSpeedMps = p.speed;
    v.T = p.headway; v.a = p.accel; v.b = p.decel;
  }
}
