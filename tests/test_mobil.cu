#define LPSIM_SCALABLE_LIBRARY
#include "../src/simulator/scalable_junction_simulator.cu"
#include <cstring>

void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
Car car(int lane, float pos, float speed, int road=0) {
    Car c{}; c.state=1; c.n=2; c.edges[0]=road; c.edges[1]=2; c.lanes[0]=lane;
    c.lanes[1]=0; c.moves[0]=road*3+lane; c.pos=pos; c.v=speed; return c;
}
int main() { try {
    check(cudaSetDevice(0));
    std::vector<Road> roads{{0,1,3,500,30},{2,3,3,500,30},{1,4,1,500,30}};
    std::vector<Movement> moves;
    for(int road=0;road<2;++road)for(int lane=0;lane<3;++lane)
        moves.push_back({roads[road].to,road,2,lane,0,0,0,20,0});
    auto* dr=device(roads);auto* dm=device(moves);mobil::Config cfg;cfg.enabled=true;
    auto run = [&](std::vector<Car> cars, mobil::Config config, const std::vector<int>& ids,
                   const std::vector<int>& owners, unsigned long long expected, const char* label) {
        auto reference=cars;
        // Reference IDs are canonical; reorder before serial evaluation.
        std::vector<Car> canonical(cars.size());
        for(int i=0;i<int(cars.size());++i)canonical[ids[i]]=cars[i];
        if(owners.empty())mobil::reference(canonical,roads,moves,config,0,2);
        auto* dc=device(cars);auto* di=device(ids);int* own=owners.empty()?nullptr:device(owners);
        mobil::Engine engine(cars.size(),roads.size(),moves,config);
        engine.step(dc,dr,dm,0,2,di,own,0);check(cudaDeviceSynchronize());
        require(engine.count()==expected,label);
        check(cudaMemcpy(cars.data(),dc,cars.size()*sizeof(Car),cudaMemcpyDeviceToHost));
        for(int i=0;i<int(cars.size());++i)require(std::memcmp(&cars[i],owners.empty()?&canonical[ids[i]]:&reference[i],sizeof(Car))==0,"GPU/reference or ownership mismatch");
        cudaFree(dc);cudaFree(di);if(own)cudaFree(own);
        std::cout<<"PASS "<<label<<'\n';
    };
    run({car(0,100,15),car(0,120,5)},cfg,{0,1},{},1,"beneficial overtaking and legal connector update");
    auto disabled=cfg;disabled.enabled=false;
    run({car(0,100,15),car(0,120,5)},disabled,{0,1},{},0,"disabled mode preserves state");
    run({car(0,100,15),car(0,120,5)},cfg,{0,1},{1,1},0,"ghosts cannot initiate changes");
    run({car(0,100,15),car(0,120,5),car(2,100,15),car(2,120,5)},cfg,{0,1,2,3},{},1,"same-road conflicts admit only one stable ID");
    run({car(2,120,5),car(2,100,15),car(0,120,5),car(0,100,15)},cfg,{3,2,1,0},{},1,"slot permutation preserves winner");
    run({car(0,100,15,0),car(0,120,5,0),car(0,100,15,1),car(0,120,5,1)},cfg,{0,1,2,3},{},2,"independent roads change concurrently");
    run({car(0,485,15),car(0,495,5)},cfg,{0,1},{},0,"junction exclusion zone");
    auto threshold=cfg;threshold.threshold=1.e6f;
    run({car(0,100,15),car(0,120,5)},threshold,{0,1},{},0,"incentive threshold");
    std::vector<Car> danger{car(0,100,15),car(0,120,5),car(1,90,30)};
    auto selfish=cfg;selfish.politeness=0;
    run(danger,selfish,{0,1,2},{},0,"GPU rejects excessive target-follower braking");
    auto p=mobil::evaluate(danger.data(),0,1,1,{1,-1},{-1,2},roads.data(),moves.data(),cfg,0,2);
    require(p.lane<0,"unsafe target follower accepted");
    danger[2]=car(1,100,5);
    p=mobil::evaluate(danger.data(),0,1,1,{1,-1},{2,-1},roads.data(),moves.data(),cfg,0,2);
    require(p.lane<0,"overlapping target accepted");
    // A higher politeness factor can reject an egoistic gain that brakes a follower.
    bool checked=false;
    for(float position=40;position<90&&!checked;position+=2) {
        danger[2]=car(1,position,18);auto selfish=cfg;selfish.politeness=0;auto polite=cfg;polite.politeness=1;
        auto a=mobil::evaluate(danger.data(),0,1,1,{1,-1},{-1,2},roads.data(),moves.data(),selfish,0,2);
        auto b=mobil::evaluate(danger.data(),0,1,1,{1,-1},{-1,2},roads.data(),moves.data(),polite,0,2);
        if(a.lane>=0 && b.lane>=0){require(b.gain<a.gain,"politeness did not charge induced braking");checked=true;}
    }
    require(checked,"politeness fixture did not exercise follower penalty");
    auto limited=moves;limited.erase(limited.begin()+1);auto legal=mobil::legalMovements(limited);
    require(legal[1]==-1,"missing legal connector accepted");
    cudaFree(dr);cudaFree(dm);
    std::cout<<"PASS MOBIL safety, overlap, politeness and connector legality\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }
