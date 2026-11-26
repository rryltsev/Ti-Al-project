// VCSGC bias around the original, unchanged ATAT CE. ATAT source/object files
// are read-only; only this driver and its binary are new project artifacts.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "mclib.h"
#include "parse.h"
#include "drawpd.h"
#include "keci.h"
#include "plugin.h"
#include "qha_volume_atat.h"
template<> GenericPlugIn<KSpaceECI> *GenericPlugIn<KSpaceECI>::list=NULL;

class BiasedMC:public MonteCarlo {
 double kap,target;
 QHATable table; Array<Real> jcur,jtemp; double vol; bool fixed_volume=false;
 double energy_for(const Array<Real>&j)const{double e=0;for(int c=0;c<total_clusters;c++)e+=rcluster_mult_per_atom[c]*cur_rho(c)*j(c);return e;}
 void apply(const Array<Real>&j){E_ref=(which_is_empty>=0?j(which_is_empty)/site_in_cell:0);for(int s=0;s<site_in_cell;s++)for(int c=0;c<nb_clusters[s];c++)eci[s][c]=j(which_cluster[s][c]);cur_energy=energy_for(j)-E_ref-mu*cur_conc;}
 long accepted;
 std::vector<int> domains;
 public:
 BiasedMC(const Structure&l,const iVector3d&s,const SpaceGroup&sg,const LinkedList<Cluster>&cl,double k,double c,const Structure&pattern):MonteCarlo(l,s,sg,cl),kap(k),target(c),accepted(0){
  rMatrix3d inv=!(pattern.cell);int population[4]={0,0,0,0};
  for(int x=0;x<supercell(0);x++)for(int y=0;y<supercell(1);y++)for(int z=0;z<supercell(2);z++)for(int b=0;b<site_in_cell;b++){
   rVector3d p=lattice.cell*to_real(iVector3d(x,y,z))+lattice.atom_pos(b);int found=-1;
   for(int d=0;d<4;d++){
    rVector3d shift=lattice.cell*to_real(iVector3d(d%2,d/2,0));
    for(int j=0;j<pattern.atom_pos.get_size();j++)if(pattern.atom_type(j)==1 && is_int(inv*(p-shift-pattern.atom_pos(j))))found=d;
   }
   if(found<0)throw std::runtime_error("D019 sublattice mapping failed");domains.push_back(found);population[found]++;
  }
  for(int d=0;d<4;d++)if(population[d]*4!=(int)rspin_size)throw std::runtime_error("Noncommensurate MC box");
 }
 void prepare_volume(){
  table.read("qha_ce.dat",total_clusters,8.617333262145e-5);
  double best=1e99;vol=table.middle();
  for(int i=0;i<201;i++){double v=table.lo()+(table.hi()-table.lo())*i/200.;Array<Real> j,jt;table.evaluate(&j,&jt,T,v);double e=energy_for(j);if(e<best){best=e;vol=v;}}
  if(std::getenv("FIXED_VOLUME_A3_ATOM")){fixed_volume=true;vol=std::atof(std::getenv("FIXED_VOLUME_A3_ATOM"));}
  table.evaluate(&jcur,&jtemp,T,vol);apply(jcur);
 }
 void volume_step(){if(fixed_volume)return;double v=vol+.035*(2*uniform01()-1);if(v<=table.lo()||v>=table.hi())return;Array<Real> j,jt;table.evaluate(&j,&jt,T,v);double de=rspin_size*(energy_for(j)-energy_for(jcur));if(de<=0||uniform01()<exp(-de/T)){vol=v;jcur=j;jtemp=jt;apply(jcur);}}
 double get_volume()const{return vol;}
 double thermodynamic_energy(){return get_cur_energy()+mu*cur_conc-(T/table.kb)*energy_for(jtemp);}
 double count()const{return rspin_size;}
 double acceptance(){double a=accepted/rspin_size;accepted=0;return a;}
 void order(double *f)const {
  int n[4]={0,0,0,0},den[4]={0,0,0,0},i=0;
  for(int x=0;x<supercell(0);x++)for(int y=0;y<supercell(1);y++)for(int z=0;z<supercell(2);z++)for(int b=0;b<site_in_cell;b++,i++){
   int s=domains[i];int offset=(((x+margin(0))*total_box(1)+(y+margin(1)))*total_box(2)+z+margin(2))*site_in_cell+b;
   n[s]+=(spin[offset]+1)/2;den[s]++;
  }
  for(int i=0;i<4;i++)f[i]=(double)n[i]/den[i];
 }
 void slab_start(double c){
  Structure s;iMatrix3d ms;ms.diag(supercell);s.cell=lattice.cell*to_real(ms);
  int N=(int)rspin_size;s.atom_pos.resize(N);s.atom_type.resize(N);int i=0;
  double lo=.07,hi=.25,frac=(c-lo)/(hi-lo);frac=std::max(0.,std::min(1.,frac));
  for(int x=0;x<supercell(0);x++)for(int y=0;y<supercell(1);y++)for(int z=0;z<supercell(2);z++)for(int b=0;b<site_in_cell;b++,i++){
   s.atom_pos(i)=lattice.cell*to_real(iVector3d(x,y,z))+lattice.atom_pos(b);
   int sub=domains[i];bool slab=(x+.5)<frac*supercell(0);
   bool al=slab ? sub==0 : uniform01()<lo;s.atom_type(i)=al ? 1:-1;
  }
  init_structure(s);set_concentration(2*c-1);
 }
 protected:
 Real extension_spin_flip_energy(int *cell,int incell,int newspin){
  double c=(cur_conc+1)/2,dc=(double)newspin/rspin_size;
  return T*kap*rspin_size*(2*(c-target)*dc+dc*dc);
 }
 void extension_update_spin_flip(int *cell,int incell,int newspin,Real delta){accepted++;}
};
int main(int argc,char**argv){
 if(argc!=13){std::cerr<<"Usage: T_K mu_spin kappa target nx ny nz eq prod seed start sample_every\n";return 2;}
 double TK=atof(argv[1]),mu=atof(argv[2]),kap=atof(argv[3]),c0=atof(argv[4]);
 int nx=atoi(argv[5]),ny=atoi(argv[6]),nz=atoi(argv[7]),eq=atoi(argv[8]),prod=atoi(argv[9]),seed=atoi(argv[10]),start=atoi(argv[11]),every=atoi(argv[12]);
 const double kb=8.617333262145e-5;rndseed(seed);
 Structure lat;Array<Arrayint> lookup;Array<AutoString> labels;rMatrix3d axes;
 {std::ifstream f("lat.in");if(!f)return 3;parse_lattice_file(&lat.cell,&lat.atom_pos,&lat.atom_type,&lookup,&labels,f,&axes);wrap_inside_cell(&lat.atom_pos,lat.atom_pos,lat.cell);}
 SpaceGroup sg;sg.cell=lat.cell;find_spacegroup(&sg.point_op,&sg.trans,lat.cell,lat.atom_pos,lat.atom_type);
 LinkedList<Cluster> clusters;{std::ifstream f("clusters.out");if(!f)return 4;read_clusters_and_eci(&clusters,NULL,f,f,axes);}
 Array<Real> J(clusters.get_size());{std::ifstream f("eci.out");if(!f)return 5;for(int i=0;i<J.get_size();i++)if(!(f>>J(i)))return 6;}
 std::vector<Structure> gs;{
  std::ifstream f("gs_str.out");
  while(!f.eof()){
   Structure s;parse_structure_file(&s.cell,&s.atom_pos,&s.atom_type,labels,f);
   if(!fix_atom_type(&s,lat,lookup,1))return 7;skip_to_next_structure(f);wrap_inside_cell(&s.atom_pos,s.atom_pos,s.cell);
   for(int i=0;i<s.atom_type.get_size();i++)s.atom_type(i)=-1+2*s.atom_type(i);gs.push_back(s);
  }
 }
 BiasedMC mc(lat,iVector3d(nx,ny,nz),sg,clusters,kap,c0,gs.at(1));mc.set_eci(J);
 if(start==2)mc.slab_start(c0);
 else if(start==1){mc.init_structure(gs.at(1));mc.set_concentration(2*c0-1);}
 else{mc.init_random(2*c0-1);mc.set_concentration(2*c0-1);}
 mc.init_run(kb*TK,mu);mc.prepare_volume();
 double op[4];mc.order(op);std::cerr<<std::setprecision(16)<<"N "<<mc.count()<<" initial_sub "<<op[0]<<" "<<op[1]<<" "<<op[2]<<" "<<op[3]<<" initial_energy "<<mc.get_cur_energy()+mu*mc.get_cur_concentration()<<"\n";
 {std::ofstream f("initial_correlations.csv");f<<std::setprecision(16);const Array<Real>&corr=mc.get_cur_corr();for(int i=0;i<corr.get_size();i++)f<<i<<","<<corr(i)<<"\n";}
 std::ofstream out("samples.csv");out<<std::setprecision(16)<<"pass,c,mu_occ_eV,E_CE_eV_atom,eta,s0,s1,s2,s3,acceptance,volume_A3_atom,U_thermo_eV_atom\n";
 for(int pass=1;pass<=eq+prod;pass++){
  mc.run(1);mc.volume_step();double acceptance=mc.acceptance();
  if(pass%every==0){
   double c=(mc.get_cur_concentration()+1)/2;mc.order(op);double maximum=op[0];for(int i=1;i<4;i++)maximum=std::max(maximum,op[i]);
   out<<pass<<","<<c<<","<<(2*mu-2*kb*TK*kap*(c-c0))<<","<<(mc.get_cur_energy()+mu*mc.get_cur_concentration())<<","<<((maximum-c)/.75);
   for(int i=0;i<4;i++)out<<","<<op[i];out<<","<<acceptance<<","<<mc.get_volume()<<","<<mc.thermodynamic_energy()<<"\n";
  }
 }
 double before=mc.get_cur_energy();mc.calc_from_scratch();std::cerr<<std::setprecision(16)<<"energy_drift "<<(mc.get_cur_energy()-before)<<"\n";
 std::ofstream f("snapshot.out");mc.view(lookup,labels,f,axes);
 return fabs(mc.get_cur_energy()-before)>1e-8 ? 8:0;
}
