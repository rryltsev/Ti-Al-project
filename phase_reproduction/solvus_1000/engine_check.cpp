#include "engine.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

// Save actual trial and accepted states for independent MLIP evaluation.
// Small cells exercise repeated periodic images and a change in the central
// species of its own images. Larger cells exercise the local cache.
struct Geometry {
    Cell cell;
    std::vector<Vec3> pos;
    std::vector<int> types;
};

void write_state(std::ofstream& cfg,std::ofstream& log,const Geometry& g,
                 std::string label,double expected) {
    cfg<<"BEGIN_CFG\n Size\n "<<g.pos.size()<<"\n Supercell\n";
    for(auto& r:g.cell)cfg<<r[0]<<" "<<r[1]<<" "<<r[2]<<"\n";
    cfg<<" AtomData: id type cartes_x cartes_y cartes_z\n";
    for(size_t i=0;i<g.pos.size();++i)
        cfg<<i+1<<" "<<g.types[i]<<" "<<g.pos[i][0]<<" "<<g.pos[i][1]<<" "<<g.pos[i][2]<<"\n";
    cfg<<"END_CFG\n\n";log<<g.pos.size()<<" "<<label<<" "<<expected<<"\n";
}

int main(int argc,char** argv){
    if(argc!=2)return 2;
    std::mt19937_64 rng(912375);
    std::uniform_real_distribution<double> u(-.12,.12);
    std::ofstream cfg("engine_cases.cfg"),log("engine_energies.txt");
    cfg<<std::setprecision(17);log<<std::setprecision(17);
    for(auto dims:std::vector<std::array<int,3>>{{2,2,1},{4,4,4},{6,6,6}}){
        Geometry s;
        double a=2.955,c=4.70,root3=std::sqrt(3.);
        s.cell={Vec3{dims[0]*a,0,0},Vec3{dims[1]*a/2,dims[1]*a*root3/2,0},Vec3{0,0,dims[2]*c}};
        for(int i=0;i<dims[0];++i)for(int j=0;j<dims[1];++j)for(int z=0;z<dims[2];++z)for(int b=0;b<2;++b){
            s.pos.push_back({a*(i+b/3.+.5*(j+b/3.))+u(rng),a*(j+b/3.)*root3/2+u(rng),c*(z+b*.5)+u(rng)});
            s.types.push_back(int(rng()%5==0));
        }
        Engine e(argv[1],s.cell,s.pos,s.types);
        write_state(cfg,log,s,"construct",e.energy());
        for(int k=0;k<20;++k){
            Geometry t=s;double de;
            std::string label;
            if(k%6==5){
                double ratio=1.+u(rng)*.05;
                for(auto& r:t.cell)for(double& v:r)v*=ratio;
                for(auto& r:t.pos)for(double& v:r)v*=ratio;
                de=e.set_volume_trial(ratio);label="volume";
            }else{
                unsigned i=rng()%s.pos.size();
                if(k%3!=0)for(double& v:t.pos[i])v+=u(rng);
                if(k%3!=1)t.types[i]=1-t.types[i];
                if(k==10)t.pos[i][0]+=.79; // Trigger the single large-move fallback.
                de=e.trial(i,t.pos[i],t.types[i]);label=k%3==0?"flip":(k%3==1?"displace":"flip+displace");
            }
            write_state(cfg,log,t,label,e.energy()+de);
            if(k%4==1)e.reject();else{e.accept();s=std::move(t);}
            write_state(cfg,log,s,"committed",e.energy());
        }
    }
}
