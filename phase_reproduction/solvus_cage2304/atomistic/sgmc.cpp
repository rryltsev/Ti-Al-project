// Semigrand Monte Carlo in the condensed, site-confined HCP basin.
#include "engine.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr double kb=8.617333262145e-5, vref=17.288329847774033;
constexpr double axial_ref=4.6046/2.9446, root3=1.7320508075688772935;
using V3=std::array<double,3>;
using Box=std::array<V3,3>;
V3 plus(V3 a,const V3& b,double s=1) { for(int j=0;j<3;++j) a[j]+=s*b[j]; return a; }
V3 stretch(V3 a,const V3& scale) { for(int j=0;j<3;++j) a[j]*=scale[j]; return a; }
double dot(const V3& a,const V3& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
struct Shape {
    double v,axial,nearest;
    Box cell;
    V3 scale;
    std::array<std::array<V3,12>,2> normals{};
    Shape(int nx,int nz,double v_,double ar):v(v_),axial(ar) {
        double a=std::cbrt(4*v/(root3*ar)),c=a*ar;
        double a0=std::cbrt(4*vref/(root3*axial_ref));
        scale={a/a0,a/a0,c/(a0*axial_ref)};
        cell={V3{nx*a,0,0},V3{nx*a/2,nx*root3*a/2,0},V3{0,0,nz*c}};
        nearest=std::min(a,std::sqrt(a*a/3+c*c/4));
        for(int b=0;b<2;++b) {
            std::vector<V3> bonds;
            for(V3 e:std::array<V3,3>{V3{a,0,0},V3{a/2,a*root3/2,0},V3{-a/2,a*root3/2,0}}) {
                bonds.push_back(e); bonds.push_back(stretch(e,{-1,-1,-1}));
            }
            double sign=b?-1.:1.;
            for(double z:{-c/2,c/2}) {
                bonds.push_back({sign*a/2,sign*a*root3/6,z});
                bonds.push_back({-sign*a/2,sign*a*root3/6,z});
                bonds.push_back({0,-sign*a*root3/3,z});
            }
            for(int i=0;i<12;++i) for(int j=0;j<3;++j)
                normals[b][i][j]=2*bonds[i][j]*scale[j]/dot(bonds[i],bonds[i]);
        }
    }
    double wall(int atom,V3 d) const {
        double w=0; for(const auto& n:normals[atom%2]) w=std::max(w,dot(n,d)); return w;
    }
};

struct Chain {
    int nx,nz,n,k=0;
    double kt,cap,mu,K,q0,wall_bound=0;
    std::string model;
    Shape shape;
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> uniform{0,1};
    std::vector<V3> ideal,u;
    V3 mean{};
    std::vector<int> types,sub;
    std::array<int,4> occ{};
    std::unique_ptr<Engine> engine;
    long long accepted_flip=0,accepted_pos=0,accepted_vol=0,accepted_ax=0,vol_count=0,ax_count=0;

    Chain(int basal,int vertical,double cap_,double mu_,double K_,double q_,unsigned long long seed,
          const std::string& start,const std::string& path,double temperature)
        :nx(basal),nz(vertical),n(2*nx*nx*nz),kt(kb*temperature),cap(cap_),mu(mu_),K(K_),q0(q_),model(path),
         shape(nx,nz,17.4,axial_ref),rng(seed),u(n,V3{}),types(n,0) {
        if(nx%2 || nx<2 || nz<2 || cap<=0 || cap>.4 || K<0 || !std::isfinite(kt) || kt<=0)
            throw std::invalid_argument("cell, temperature or cage");
        double a=std::cbrt(4*vref/(root3*axial_ref));
        for(int i=0;i<nx;++i) for(int j=0;j<nx;++j) for(int z=0;z<nz;++z) for(int b=0;b<2;++b) {
            double h=i+b/3., l=j+b/3.;
            ideal.push_back({a*(h+l*.5),a*root3*.5*l,(z+b*.5)*a*axial_ref});
            sub.push_back(2*((i+b)%2)+(j+b)%2);
        }
        if(start=="ordered" || start=="random" || start=="window") {
            double qo=start=="ordered"?.81:(start=="random"?0.:std::clamp(q0,0.,.93));
            double x=.145+.10*qo;
            int preferred=std::uniform_int_distribution<int>(0,3)(rng);
            for(int g=0;g<4;++g) {
                std::vector<int> sites; for(int i=0;i<n;++i) if(sub[i]==g) sites.push_back(i);
                std::shuffle(sites.begin(),sites.end(),rng);
                int m=std::clamp(int(std::round((x+(g==preferred?.75:-.25)*qo)*sites.size())),0,(int)sites.size());
                for(int i=0;i<m;++i) types[sites[i]]=1;
            }
        } else {
            std::ifstream f(start); int m; double v,ar;
            if(!(f>>m>>v>>ar) || m!=n) throw std::runtime_error("bad restart");
            shape=Shape(nx,nz,v,ar);
            for(int i=0;i<n;++i) {
                if(!(f>>types[i]>>u[i][0]>>u[i][1]>>u[i][2]) || (types[i]!=0 && types[i]!=1))
                    throw std::runtime_error("bad restart rows");
                mean=plus(mean,u[i],1./n);
            }
        }
        for(int i=0;i<n;++i){ k+=types[i]; occ[sub[i]]+=types[i]; }
        wall_bound=full_wall(shape);
        if(wall_bound>=cap) throw std::runtime_error("restart outside cage");
        engine=build(shape);
    }
    V3 pos(int i,const Shape& s) const {return stretch(plus(ideal[i],u[i]),s.scale);}
    std::unique_ptr<Engine> build(const Shape& s) const {
        std::vector<V3> positions(n); for(int i=0;i<n;++i)positions[i]=pos(i,s);
        return std::make_unique<Engine>(model,s.cell,positions,types);
    }
    double full_wall(const Shape& s) const {
        double w=0; for(int i=0;i<n;++i) w=std::max(w,s.wall(i,plus(u[i],mean,-1)));
        return w;
    }
    double order() const {return 4.*(4*(*std::max_element(occ.begin(),occ.end()))-k)/(3*n);}
    double bias() const {double d=order()-q0; return .5*K*d*d;}
    bool take(double dimensionless_delta) {return dimensionless_delta<=0 || std::log(uniform(rng)) < -dimensionless_delta;}
    double sym(double limit) {return (2*uniform(rng)-1)*limit;}
    int random_site() {return std::uniform_int_distribution<int>(0,n-1)(rng);}
    void flip() {
        int i=random_site(),change=1-2*types[i];
        double old=bias();
        k+=change; occ[sub[i]]+=change;
        double de=engine->trial(i,pos(i,shape),1-types[i]);
        if(take((de-mu*change+bias()-old)/kt)) {
            engine->accept(); types[i]=1-types[i]; ++accepted_flip;
        }else{engine->reject(); k-=change; occ[sub[i]]-=change;}
    }
    void displace() {
        int i=random_site();
        V3 du={sym(.13)/shape.scale[0],sym(.13)/shape.scale[1],sym(.13)/shape.scale[2]};
        V3 saved=u[i],oldmean=mean;
        u[i]=plus(u[i],du); mean=plus(mean,du,1./n);
        double wi=shape.wall(i,plus(u[i],mean,-1));
        if(wi>=cap){u[i]=saved;mean=oldmean;return;}
        V3 change=stretch(du,shape.scale);
        double ub=std::max(wi,wall_bound+2*std::sqrt(dot(change,change))/(n*shape.nearest));
        if(ub>=cap) ub=full_wall(shape);
        if(ub>=cap){u[i]=saved;mean=oldmean;return;}
        double de=engine->trial(i,pos(i,shape),types[i]);
        if(take(de/kt)){engine->accept();wall_bound=ub;++accepted_pos;}
        else {engine->reject();u[i]=saved;mean=oldmean;}
    }
    void volume() {
        ++vol_count;
        double v=shape.v+sym(.22*std::sqrt(128./n));
        if(v<15. || v>21.)return;
        Shape next(nx,nz,v,shape.axial);
        if(next.nearest*(1-cap)<1.68052)return;
        double de=engine->set_volume_trial(std::cbrt(v/shape.v));
        // Symmetric in V. N-1 scaled relative coordinates and one restored
        // translational factor V. The imposed cage follows the same lattice.
        if(take(de/kt-n*std::log(v/shape.v))){engine->accept(); shape=next;++accepted_vol;}
        else engine->reject();
    }
    void axial() {
        ++ax_count;
        double ar=shape.axial*std::exp(sym(.008*std::sqrt(128./n)));
        if(ar<1.45 || ar>1.72)return;
        Shape next(nx,nz,shape.v,ar);
        double w=full_wall(next);
        if(w>=cap || next.nearest*(1-cap)<1.68052)return;
        auto proposed=build(next);
        if(take((proposed->energy()-engine->energy())/kt)) {
            engine=std::move(proposed);shape=next;wall_bound=w;++accepted_ax;
        }
    }
    void snapshot(const std::string& path) const {
        std::ofstream f(path); f<<std::setprecision(17)<<n<<" "<<shape.v<<" "<<shape.axial<<"\n";
        for(int i=0;i<n;++i){V3 d=plus(u[i],mean,-1);f<<types[i]<<" "<<d[0]<<" "<<d[1]<<" "<<d[2]<<"\n";}
        if(!f)throw std::runtime_error("checkpoint write");
    }
};
}

int main(int argc,char** argv) {
    if(argc!=16) {std::cerr<<"sgmc model nx nz cap mu K q0 seed start burn sweeps stride axial_free prefix temperature_K\n";return 2;}
    try {
        const int burn=std::stoi(argv[10]),prod=std::stoi(argv[11]),stride=std::stoi(argv[12]);
        bool free_axial=std::stoi(argv[13])!=0;
        std::string prefix=argv[14];
        const double temperature=std::stod(argv[15]);
        Chain a(std::stoi(argv[2]),std::stoi(argv[3]),std::stod(argv[4]),std::stod(argv[5]),
            std::stod(argv[6]),std::stod(argv[7]),std::stoull(argv[8]),argv[9],argv[1],temperature);
        std::ofstream f(prefix+".csv"); f<<std::setprecision(12)
            <<"sweep,n_Al,n0,n1,n2,n3,Q,volume_A3_atom,E_eV,wall,c_a,acc_flip,acc_position,acc_volume,acc_axial,T_K\n";
        for(int sweep=1;sweep<=burn+prod;++sweep) {
            for(int i=0;i<a.n;++i){a.displace();a.flip();}
            a.volume(); if(free_axial && sweep%3==0)a.axial();
            if(sweep%stride==0){
                a.wall_bound=a.full_wall(a.shape);
                if(a.wall_bound>=a.cap)throw std::runtime_error("wall verification failed");
                f<<sweep<<","<<a.k;
                for(int o:a.occ)f<<","<<o;
                f<<","<<a.order()<<","<<a.shape.v<<","<<a.engine->energy()<<","<<a.wall_bound
                 <<","<<a.shape.axial<<","<<a.accepted_flip/double(sweep*a.n)<<","<<a.accepted_pos/double(sweep*a.n)
                 <<","<<a.accepted_vol/double(a.vol_count)<<","<<(a.ax_count?a.accepted_ax/double(a.ax_count):0)<<","<<temperature<<"\n";
            }
            if(sweep%50==0){f.flush();a.snapshot(prefix+".state");}
        }
        a.snapshot(prefix+".state");
        auto exact=a.build(a.shape);
        double drift=std::abs(exact->energy()-a.engine->energy());
        if(!(drift<1e-6))throw std::runtime_error("final energy drift "+std::to_string(drift));
        if(!f)throw std::runtime_error("trace write");
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
