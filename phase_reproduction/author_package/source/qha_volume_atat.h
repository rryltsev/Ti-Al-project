// Isolated ATAT extension: joint chemical and scalar-volume Monte Carlo.
#ifndef TIAL_QHA_VOLUME_H
#define TIAL_QHA_VOLUME_H
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include "mclib.h"
#include "drawpd.h"

class QHATable {
public:
  int nv,nt,nc;
  double kb;
  std::vector<double> volumes,temperatures,coefficients;
  QHATable():nv(0),nt(0),nc(0),kb(0) {}
  void read(const char *filename,int expected,double boltzmann) {
    std::ifstream f(filename);std::string magic;
    f>>magic>>nv>>nt>>nc>>kb;
    if(!f || magic!="QHA_CE_V1" || nc!=expected || nv<5 || nt<2 || std::fabs(kb-boltzmann)>1e-10)
      throw std::runtime_error("Invalid volume CE table or units");
    volumes.resize(nv);temperatures.resize(nt);
    for(int i=0;i<nv;i++)f>>volumes[i];
    for(int i=0;i<nt;i++)f>>temperatures[i];
    coefficients.resize(nt*(nv-1)*nc*8);
    for(size_t i=0;i<coefficients.size();i++)f>>coefficients[i];
    if(!f)throw std::runtime_error("Truncated volume CE table");
    for(int i=1;i<nv;i++)if(volumes[i]<=volumes[i-1])throw std::runtime_error("Unordered volumes");
  }
  double lo()const{return volumes.front();}
  double hi()const{return volumes.back();}
  double middle()const{return (lo()+hi())/2;}
  void evaluate(Array<Real> *J,Array<Real> *JT,double Tenergy,double v)const {
    double TK=Tenergy/kb;
    if(TK < -1e-8 || TK>temperatures.back()+1e-8 || v<lo() || v>hi())
      throw std::runtime_error("QHA interpolation outside calculated volume/temperature domain");
    int it=0,iv=0;
    while(it<nt-2 && TK>temperatures[it+1])it++;
    while(iv<nv-2 && v>volumes[iv+1])iv++;
    double dt=temperatures[it+1]-temperatures[it],u=(TK-temperatures[it])/dt,x=v-volumes[iv];
    double H[4]={2*u*u*u-3*u*u+1,(u*u*u-2*u*u+u)*dt,-2*u*u*u+3*u*u,(u*u*u-u*u)*dt};
    double D[4]={(6*u*u-6*u)/dt,3*u*u-4*u+1,(-6*u*u+6*u)/dt,3*u*u-2*u};
    J->resize(nc);JT->resize(nc);
    for(int c=0;c<nc;c++) {
      double jj=0,dj=0;
      for(int power=3;power>=0;power--) {
        size_t a=((it*(nv-1)+iv)*nc+c)*8;
        size_t b=(((it+1)*(nv-1)+iv)*nc+c)*8;
        double vals[4]={coefficients[a+power],coefficients[a+4+power],coefficients[b+power],coefficients[b+4+power]};
        double q=0,d=0;for(int k=0;k<4;k++){q+=H[k]*vals[k];d+=D[k]*vals[k];}
        jj=jj*x+q;dj=dj*x+d;
      }
      (*J)(c)=jj;(*JT)(c)=dj;
    }
  }
};

class QHAMonteCarlo:public MonteCarlo {
  const QHATable *table;
  Array<Real> currentJ,currentJT;
  double v,lastT,step;
  bool initialized;
  long proposals,accepted;
  std::vector<int> domains;
  int population[4];
  double energy_for(const Array<Real>&j)const {
    double e=0;
    for(int c=0;c<total_clusters;c++)e+=rcluster_mult_per_atom[c]*cur_rho(c)*j(c);
    return e;
  }
  void apply(const Array<Real>&j) {
    E_ref=(which_is_empty>=0 ? j(which_is_empty)/site_in_cell : 0);
    for(int s=0;s<site_in_cell;s++)for(int c=0;c<nb_clusters[s];c++)eci[s][c]=j(which_cluster[s][c]);
    cur_energy=energy_for(j)-E_ref-mu*cur_conc;
  }
  double objective(double vv)const {
    Array<Real> j,jt;table->evaluate(&j,&jt,T,vv);return energy_for(j);
  }
  void initial_minimum() {
    int best=0;double bestf=objective(table->volumes[0]);
    for(int i=1;i<table->nv;i++){double f=objective(table->volumes[i]);if(f<bestf){best=i;bestf=f;}}
    double a=table->volumes[best>0 ? best-1 : 0],b=table->volumes[best<table->nv-1 ? best+1 : best];
    for(int i=0;i<40;i++){double l=a+(b-a)*.38196601125,r=a+(b-a)*.61803398875;if(objective(l)<objective(r))b=r;else a=l;}
    v=(a+b)/2;
  }
public:
  QHAMonteCarlo(const Structure&l,const iVector3d&s,const SpaceGroup&sg,const LinkedList<Cluster>&cl,const QHATable*t,const Structure *pattern=NULL):
    MonteCarlo(l,s,sg,cl),table(t),v(t->middle()),lastT(-1),step(.035),initialized(false),proposals(0),accepted(0) {
    for(int d=0;d<4;d++)population[d]=0;
    if(pattern) {
      rMatrix3d inv=!(pattern->cell);
      for(int a=0;a<supercell(0);a++)for(int b=0;b<supercell(1);b++)for(int c=0;c<supercell(2);c++)for(int s=0;s<site_in_cell;s++){
        rVector3d p=lattice.cell*to_real(iVector3d(a,b,c))+lattice.atom_pos(s);
        int found=-1;
        for(int d=0;d<4;d++) {
          rVector3d shift=lattice.cell*to_real(iVector3d(d%2,d/2,0));
          for(int j=0;j<pattern->atom_pos.get_size();j++)if(pattern->atom_type(j)==1 && is_int(inv*(p-shift-pattern->atom_pos(j))))found=d;
        }
        if(found<0)throw std::runtime_error("Cannot map site to D019 sublattice");
        domains.push_back(found);population[found]++;
      }
      for(int d=0;d<4;d++)if(population[d]*4!=(int)rspin_size)throw std::runtime_error("MC cell is not commensurate with D019 order");
    }
  }
  virtual void prepare_run() {
    if(!initialized){initial_minimum();initialized=true;}
    if(T!=lastT || currentJ.get_size()==0){table->evaluate(&currentJ,&currentJT,T,v);lastT=T;}
    apply(currentJ); // phb may have installed its ancillary fixed-volume ECI before this call.
  }
  virtual void volume_step() {
    double next=v+step*(2*uniform01()-1);proposals++;
    if(next<=table->lo() || next>=table->hi())return;
    Array<Real> j,jt;table->evaluate(&j,&jt,T,next);
    double delta=rspin_size*(energy_for(j)-energy_for(currentJ));
    if(delta<=0 || uniform01()<std::exp(-delta/T)){
      v=next;currentJ=j;currentJT=jt;apply(currentJ);accepted++;
    }
  }
  virtual Real get_cur_thermo_energy() {
    // E_eff - T_K dE_eff/dT_K - mu*x; joint volume/correlation covariance retained.
    return get_cur_energy()-(T/table->kb)*energy_for(currentJT);
  }
  virtual Real get_cur_volume()const{return v;}
  virtual Real get_cur_order_parameter()const {
    if(domains.empty())return 1.-get_cur_disorder_param();
    int counts[4]={0,0,0,0},index=0;
    for(int a=0;a<supercell(0);a++)for(int b=0;b<supercell(1);b++)for(int c=0;c<supercell(2);c++)for(int s=0;s<site_in_cell;s++,index++){
      int offset=(((a+margin(0))*total_box(1)+(b+margin(1)))*total_box(2)+(c+margin(2)))*site_in_cell+s;
      if(spin[offset]==1)counts[domains[index]]++;
    }
    double largest=0,average=0;
    for(int d=0;d<4;d++){double p=double(counts[d])/population[d];if(p>largest)largest=p;average+=p/4;}
    return 4./3*(largest-average);
  }
  double acceptance()const{return proposals ? double(accepted)/proposals : 0;}
  void reset_stats(){proposals=accepted=0;}
  double spin_count()const{return rspin_size;}
  double reweight_check() {
    double before=get_cur_energy();calc_from_scratch();return get_cur_energy()-before;
  }
};

class QHAPhase:public PhaseThermoData {
  std::string label;
  Structure lattice,str;
  SpaceGroup group;
  LinkedList<ArrayrVector3d> clusters;
  const QHATable *table;
  double lastT,lastmu,volume;
  bool cached;
  double calculate(double T,double mu,double v,double *x)const {
    Array<Real> j,jt;table->evaluate(&j,&jt,T,v);
    LinkedList<Real> list;list<<j;
    if(label=="LTE"){
      LTE_ThermoData phase(lattice,group,str,clusters,list);phase.set_T_mu(T,mu);*x=phase.x();return phase.phi();
    }
    MF_ThermoData phase(lattice,group,str,clusters,list);phase.set_T_mu(T,mu);*x=phase.x();return phase.phi();
  }
public:
  QHAPhase(const char*lab,const Structure&l,const SpaceGroup&g,const Structure&s,
           const LinkedList<ArrayrVector3d>&cl,const QHATable*t):
    label(lab),lattice(l),str(s),group(g),clusters(cl),table(t),lastT(-1),lastmu(0),volume(t->middle()),cached(false) {}
  virtual void set_T_mu(Real T,Real mu) {
    if(cached && T==lastT && mu==lastmu)return;
    double x=0,bestf=MAXFLOAT;int best=0;
    for(int i=0;i<table->nv;i++){
      double f=calculate(T,mu,table->volumes[i],&x);if(f<bestf){best=i;bestf=f;}
    }
    double a=table->volumes[best>0 ? best-1 : 0],b=table->volumes[best<table->nv-1 ? best+1 : best];
    for(int i=0;i<36;i++){
      double l=a+(b-a)*.38196601125,r=a+(b-a)*.61803398875;
      if(calculate(T,mu,l,&x)<calculate(T,mu,r,&x))b=r;else a=l;
    }
    volume=(a+b)/2;cur_phi=calculate(T,mu,volume,&x);cur_x=x;
    cached=true;lastT=T;lastmu=mu;
  }
  double get_volume()const{return volume;}
};
#endif
