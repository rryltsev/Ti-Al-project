// Single-window CPU variance-constrained semigrand MC for a binary HCP CE.
// Build: c++ -O3 -std=c++11 vcs.cpp -o vcs
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using JRow = std::array<double,20>;
const double kB = 8.617333262145e-5;
struct Motif { int orbit, degree; std::array<std::array<int,4>,3> offsets; };
struct Cluster { int a,b,c,orbit; }; // c=-1 for pairs
struct Pair { int j,orbit; double J; };
struct Triplet { int j,k,orbit; double J; };
struct Neighbors { std::vector<Pair> pairs; std::vector<Triplet> triples; };

// A single row is static or HA. A volume grid uses natural cubic splines,
// a flat dv measure and a symmetric bounded-domain Metropolis proposal at P=0.
struct Grid {
    std::vector<double> v;
    std::vector<JRow> y, second;
    explicit Grid(const std::string &path) {
        std::ifstream f(path);
        // Optional text header: each J row also contains its exact second
        // derivatives in volume (e.g. an author's not-a-knot spline).
        const bool supplied=(f>>std::ws).peek()=='#';
        if(supplied) {
            std::string header;
            std::getline(f,header);
            if(!header.empty() && header.back()=='\r') header.pop_back();
            if(header!="# volume J[20] d2J_dv2[20]") throw std::runtime_error("invalid spline header");
        }
        double volume;
        while (f >> volume) {
            if (!std::isfinite(volume) || volume <= 0 || (!v.empty() && volume <= v.back()))
                throw std::runtime_error("invalid volume");
            JRow row;
            for (double &j:row) if (!(f>>j) || !std::isfinite(j)) throw std::runtime_error("invalid coefficients");
            v.push_back(volume); y.push_back(row);
            if(supplied) {
                JRow d2;
                for(double &x:d2) if(!(f>>x) || !std::isfinite(x)) throw std::runtime_error("invalid spline curvature");
                second.push_back(d2);
            }
        }
        if (v.empty() || !f.eof()) throw std::runtime_error("invalid coefficient file");
        if(supplied) return;
        second.resize(v.size(), JRow{});
        std::vector<double> upper(v.size());
        for (size_t i=1; i+1<v.size(); ++i) {
            double left=v[i]-v[i-1], right=v[i+1]-v[i];
            double diag=2*(left+right)-left*upper[i-1];
            upper[i]=right/diag;
            for (size_t j=0;j<20;++j) {
                double rhs=6*((y[i+1][j]-y[i][j])/right-(y[i][j]-y[i-1][j])/left);
                second[i][j]=(rhs-left*second[i-1][j])/diag;
            }
        }
        for (int i=int(v.size())-2;i>=1;--i)
            for (size_t j=0;j<20;++j) second[i][j]-=upper[i]*second[i+1][j];
    }
    JRow at(double x) const {
        if (v.size()==1) return y[0];
        if (x<v.front() || x>v.back()) throw std::runtime_error("volume outside grid");
        size_t i=std::min<size_t>(v.size()-2,std::upper_bound(v.begin(),v.end(),x)-v.begin()-1);
        double h=v[i+1]-v[i], a=(v[i+1]-x)/h, b=1-a;
        JRow row;
        for (size_t j=0;j<20;++j)
            row[j]=a*y[i][j]+b*y[i+1][j]+h*h/6*((a*a*a-a)*second[i][j]+(b*b*b-b)*second[i+1][j]);
        return row;
    }
};

struct Lattice {
    int n;
    std::vector<int> s,color;
    std::vector<Cluster> clusters;
    std::vector<Neighbors> adj;
    std::vector<double> field;
    std::vector<std::array<int,20>> orbit_field;
    Lattice(int nx,int ny,int nz,const std::string &path):n(2*nx*ny*nz),s(n,-1),color(n),adj(n),field(n) {
        std::ifstream f(path);
        std::vector<Motif> motifs;
        Motif m{};
        while(f>>m.orbit>>m.degree) {
            if(m.orbit<2 || m.orbit>=20 || (m.degree!=2 && m.degree!=3)) throw std::runtime_error("bad motif");
            for(int k=0;k<m.degree;++k) for(int &x:m.offsets[k])
                if(!(f>>x)) throw std::runtime_error("truncated motif");
            motifs.push_back(m);
        }
        if(motifs.empty() || !f.eof()) throw std::runtime_error("unreadable motifs");
        auto wrap=[](int x,int size){return (x%size+size)%size;};
        for(int i=0;i<nx;++i) for(int j=0;j<ny;++j) for(int k=0;k<nz;++k) {
            for(int b=0;b<2;++b) color[2*((i*ny+j)*nz+k)+b]=2*((i+b)%2)+(j+b)%2;
            for(const Motif &t:motifs) {
                int site[3]={-1,-1,-1};
                for(int l=0;l<t.degree;++l) {
                    auto off=t.offsets[l];
                    site[l]=2*((wrap(i+off[0],nx)*ny+wrap(j+off[1],ny))*nz+wrap(k+off[2],nz))+off[3];
                }
                if(site[0]==site[1] || site[1]==site[2] || site[0]==site[2])
                    throw std::runtime_error("periodic image aliases in selected MC shape");
                clusters.push_back({site[0],site[1],site[2],t.orbit});
            }
        }
    }
    JRow sums() const {
        JRow sums{};
        sums[0]=n*0.5; sums[1]=std::accumulate(s.begin(),s.end(),0);
        for (const auto &t:clusters)
            sums[t.orbit]+=s[t.a]*s[t.b]*(t.c<0 ? 1:s[t.c]);
        return sums;
    }
    void refresh(const JRow &J, bool variable_volume=false) {
        if(variable_volume) orbit_field.assign(n, std::array<int,20>{});
        std::fill(field.begin(),field.end(),J[1]);
        for(auto &a:adj) {a.pairs.clear(); a.triples.clear();}
        for(const auto &t:clusters) {
            const int a=t.a,b=t.b,c=t.c;
            double j=J[t.orbit];
            if(c<0) {
                adj[a].pairs.push_back({b,t.orbit,j}); adj[b].pairs.push_back({a,t.orbit,j});
                field[a]+=j*s[b]; field[b]+=j*s[a];
                if(variable_volume) {
                    orbit_field[a][t.orbit]+=s[b]; orbit_field[b][t.orbit]+=s[a];
                }
            } else {
                adj[a].triples.push_back({b,c,t.orbit,j}); adj[b].triples.push_back({a,c,t.orbit,j});
                adj[c].triples.push_back({a,b,t.orbit,j});
                field[a]+=j*s[b]*s[c]; field[b]+=j*s[a]*s[c]; field[c]+=j*s[a]*s[b];
                if(variable_volume) {
                    orbit_field[a][t.orbit]+=s[b]*s[c]; orbit_field[b][t.orbit]+=s[a]*s[c];
                    orbit_field[c][t.orbit]+=s[a]*s[b];
                }
            }
        }
    }
    void flip(int i) {
        s[i]=-s[i];
        int ds=2*s[i];
        for(const auto &p:adj[i].pairs) field[p.j]+=ds*p.J;
        for(const auto &t:adj[i].triples) {
            double dj=ds*t.J;
            field[t.j]+=dj*s[t.k]; field[t.k]+=dj*s[t.j];
        }
    }
    double local_field(int i,const JRow &J) const {
        return std::inner_product(orbit_field[i].begin()+2,orbit_field[i].end(),J.begin()+2,J[1]);
    }
    void flip_volume_grid(int i,JRow &sums) {
        s[i]=-s[i];
        int ds=2*s[i];
        sums[1]+=ds;
        for(size_t j=2;j<sums.size();++j) sums[j]+=ds*orbit_field[i][j];
        for(const auto &p:adj[i].pairs) orbit_field[p.j][p.orbit]+=ds;
        for(const auto &t:adj[i].triples) {
            orbit_field[t.j][t.orbit]+=ds*s[t.k]; orbit_field[t.k][t.orbit]+=ds*s[t.j];
        }
    }
};

double dot(const JRow &a,const JRow &b) {return std::inner_product(a.begin(),a.end(),b.begin(),0.);}

int main(int argc,char **argv) {
  try {
    if(argc!=18 && (argc!=19 || std::string(argv[18])!="orbit-sums"))
        throw std::runtime_error("vcs motifs coeff nx ny nz T c0 kappa delta_mu equil prod stride seed random|ordered model volume_step out.csv [orbit-sums]");
    const bool record_orbits=argc==19;
    int nx=std::stoi(argv[3]),ny=std::stoi(argv[4]),nz=std::stoi(argv[5]);
    double T=std::stod(argv[6]),c0=std::stod(argv[7]),kappa=std::stod(argv[8]),mu=std::stod(argv[9]);
    int equil=std::stoi(argv[10]),prod=std::stoi(argv[11]),stride=std::stoi(argv[12]);
    auto seed=std::stoull(argv[13]);
    std::string init=argv[14], model=argv[15];
    double step=std::stod(argv[16]);
    if(nx<2 || ny<2 || nz<2 || nx%2 || ny%2 || !(T>0) || !(c0>=0 && c0<=1)
       || !(kappa>0) || !std::isfinite(mu) || equil<0 || prod<1 || stride<1 || prod%stride
       || (init!="random" && init!="ordered") || step<0 || model.find_first_of(",\n")!=std::string::npos)
        throw std::runtime_error("invalid window parameters");
    Grid grid(argv[2]);
    if(grid.v.size()>1 && !(step>0)) throw std::runtime_error("positive proposal in A^3/atom required for QHA");
    Lattice lat(nx,ny,nz,argv[1]);
    double v=std::min(grid.v.back(),std::max(grid.v.front(),17.288329847774033)), kt=kB*T;
    JRow J=grid.at(v);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform;
    std::uniform_int_distribution<int> site(0,lat.n-1);
    std::vector<int> order(lat.n); std::iota(order.begin(),order.end(),0);
    std::shuffle(order.begin(),order.end(),rng);
    if(init=="ordered") std::stable_partition(order.begin(),order.end(),[&](int i){return lat.color[i]==0;});
    int nAl=int(std::round(lat.n*c0));
    std::array<int,4> by_color{};
    for(int i=0;i<nAl;++i) {lat.s[order[i]]=1; ++by_color[lat.color[order[i]]];}
    bool variable_volume=grid.v.size()>1;
    lat.refresh(J,variable_volume);
    JRow totals=lat.sums();
    double energy=dot(totals,J);
    std::ofstream out(argv[17]);
    if(!out) throw std::runtime_error("cannot create output");
    out<<std::setprecision(12)<<"model,T_K,N,c0,kappa,delta_mu0_eV,replica,sweep,n_Al,m,volume_A3_atom,Heff_eV_atom";
    if(record_orbits)
        out<<",S_p1,S_p2,S_p3,S_p4,S_p5,S_p6,S_p7,S_t112,S_t114,S_t117,S_t123,S_t125,S_t136,S_t222_1,S_t222_2,S_t226,S_t247,S_t334";
    out<<'\n';
    uint64_t accepted=0, accepted_v=0;
    for(int sweep=1;sweep<=equil+prod;++sweep) {
        for(int attempt=0;attempt<lat.n;++attempt) {
            int i=site(rng), dn=-lat.s[i];
            double de=2*dn*(variable_volume ? lat.local_field(i,J):lat.field[i]);
            double dw=de-mu*dn+kt*kappa*(2*dn*(double(nAl)/lat.n-c0)+1./lat.n);
            if(dw<=0 || uniform(rng)<std::exp(-dw/kt)) {
                if(variable_volume) lat.flip_volume_grid(i,totals); else lat.flip(i);
                nAl+=dn; by_color[lat.color[i]]+=dn; energy+=de; ++accepted;
            }
        }
        if(grid.v.size()>1) {
            double next=v+step*(2*uniform(rng)-1);
            if(next>=grid.v.front() && next<=grid.v.back()) {
                JRow coeff=grid.at(next);
                double de=dot(totals,coeff)-dot(totals,J);
                if(de<=0 || uniform(rng)<std::exp(-de/kt)) {
                    v=next; J=coeff; energy=dot(totals,J); ++accepted_v;
                }
            }
        }
        if(sweep>equil && (sweep-equil)%stride==0) {
            int maxAl=*std::max_element(by_color.begin(),by_color.end());
            double m=4.*(4*maxAl-nAl)/(3.*lat.n);
            out<<model<<','<<T<<','<<lat.n<<','<<c0<<','<<kappa<<','<<mu<<','<<seed<<','<<sweep<<','<<nAl<<','<<m<<','<<v<<','<<energy/lat.n;
            if(record_orbits) {
                const JRow sums=variable_volume ? totals : lat.sums();
                if(std::abs(dot(sums,J)-energy)>1e-7*lat.n)
                    throw std::runtime_error("sample orbit sums disagree with incremental energy");
                for(size_t j=2;j<sums.size();++j) out<<','<<static_cast<long long>(sums[j]);
            }
            out<<'\n';
        }
    }
    JRow exact=lat.sums();
    double drift=energy-dot(exact,J);
    if(variable_volume && exact!=totals) throw std::runtime_error("cluster count drift");
    auto old=lat.field;
    if(variable_volume) for(int i=0;i<lat.n;++i) old[i]=lat.local_field(i,J);
    lat.refresh(J);
    double field_drift=0.;
    for(int i=0;i<lat.n;++i) field_drift=std::max(field_drift,std::abs(old[i]-lat.field[i]));
    if(std::abs(drift)>1e-7*lat.n || field_drift>1e-7 || !out)
        throw std::runtime_error("energy/field conservation or output failed");
    std::cout<<"samples "<<prod/stride<<" accepted "<<accepted<<" accepted_volume "<<accepted_v
             <<" drift_eV "<<drift<<" max_field_error_eV "<<field_drift<<"\n";
  } catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
