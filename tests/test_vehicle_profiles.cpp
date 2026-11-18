#include "vehicle_profiles.h"
#include <cassert>
#include <fstream>
#include <cstdio>
int main(int argc, char** argv) {
  assert(argc == 2);
  std::vector<LC::B18TrafficVehicle> v(2);
  for (int i=0;i<2;++i) {v[i].id=i;v[i].a=1.5f;v[i].b=2.5f;v[i].T=1.5f;}
  unsetenv("LPSIM_VEHICLE_PROFILES"); applyVehicleProfiles(v);
  assert(v[0].minimumGapM == -1 && v[0].a == 1.5f);
  setenv("LPSIM_VEHICLE_PROFILES", argv[1], 1);
  const char* header="vehicle_id,type_id,min_gap_m,time_headway_s,max_speed_mps,accel_mps2,decel_mps2\n";
  {std::ofstream f(argv[1]); f<<header<<"1,2,5,2.5,12,0.8,1.5\n";}
  applyVehicleProfiles(v);
  assert(v[0].a==1.5f && v[1].a==0.8f && v[1].T==2.5f && v[1].b==1.5f);
  assert(v[1].vehicleType==2 && v[1].minimumGapM==5 && v[1].desiredSpeedMps==12);
  for (const auto& bad : {"1,1,2,0,12,1,2\n", "999,1,2,1,12,1,2\n", "1,1,2,1,12,1,2\n1,1,2,1,12,1,2\n", "1,1,2,1,nan,1,2\n"}) {
    {std::ofstream f(argv[1]);f<<header<<bad;}
    bool rejected=false;try {applyVehicleProfiles(v);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected);
    assert(v[1].a==0.8f); // validation is atomic
  }
}
