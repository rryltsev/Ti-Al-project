#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Vec3 = std::array<double,3>;
using Cell = std::array<Vec3,3>;

// Native energy-only evaluator for the published two-species Ti-Al MTP.
// Basis and coefficients are taken from the provided potential file.
namespace qi_mtp {
constexpr int B=421, M=2621, K=220, L=4;
using Basic=std::array<double,B>;
using Wide=std::array<double,L>;
inline Vec3 sub(const Vec3& x,const Vec3& y){return {x[0]-y[0],x[1]-y[1],x[2]-y[2]};}
inline Vec3 mul(const Vec3& x,double s){return {s*x[0],s*x[1],s*x[2]};}
inline double dot(const Vec3& x,const Vec3& y){return x[0]*y[0]+x[1]*y[1]+x[2]*y[2];}
inline Vec3 cross(const Vec3& a,const Vec3& b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline std::vector<double> numbers(const std::string& text){
    std::vector<double> v;
    const char *p=text.c_str(); char* end=nullptr;
    while (*p) {
        const double d=std::strtod(p,&end);
        if (end!=p){v.push_back(d);p=end;} else ++p;
    }
    return v;
}
struct Product {unsigned short left,right; double coeff;};
struct Model {
    double rc=0,rmin=0,scaling=0;
    double radial[2][2][6][8]{};
    std::array<int,7> start{};
    std::array<int,K> parent{},axis{};
    std::array<int,M+1> product_start{};
    std::vector<Product> products;
    std::vector<int> mapping;
    std::vector<double> coef,offset;
    std::vector<Wide> contractions;

    explicit Model(const std::string& file):contractions(M) {
        std::ifstream f(file); if(!f)throw std::runtime_error("Cannot open MTP "+file);
        std::vector<std::string> lines; std::string l;
        while(std::getline(f,l))lines.push_back(l);
        auto field=[&](const std::string& name) {
            for(auto& s:lines){auto k=s.find_first_not_of(" \t");if(k!=std::string::npos && s.compare(k,name.size()+3,name+" = ")==0)return s.substr(k+name.size()+3);}
            throw std::runtime_error("Missing MTP field "+name);
        };
        check_dimensions(field);
        rc=std::stod(field("max_dist"));rmin=std::stod(field("min_dist"));scaling=std::stod(field("scaling"));
        for(int i=0;i<2;++i)for(int j=0;j<2;++j){
            std::string id=std::to_string(i)+"-"+std::to_string(j);
            size_t k=0;
            for(;k<lines.size();++k){
                const size_t first=lines[k].find_first_not_of(" \t\r");
                if(first!=std::string::npos && lines[k].compare(first,id.size(),id)==0)break;
            }
            if(k+6>=lines.size())throw std::runtime_error("Missing radial block "+id);
            for(int m=0;m<6;++m){
                auto v=numbers(lines[k+m+1]);if(v.size()!=8)throw std::runtime_error("Radial shape");
                std::copy(v.begin(),v.end(),radial[i][j][m]);
            }
        }
        auto m=numbers(field("alpha_index_basic"));
        if(m.size()!=B*4)throw std::runtime_error("Basic moments dimension");
        std::array<std::array<int,3>,K> powers{};
        for(int i=0;i<B;++i){
            int channel=int(m[4*i]); if(channel<0 || channel>=6)throw std::runtime_error("Radial channel");
            ++start[channel+1];
        }
        std::partial_sum(start.begin(),start.end(),start.begin());
        if(start[1]!=K)throw std::runtime_error("Monomial dimension");
        for(int mu=0;mu<6;++mu)for(int k=0;k<start[mu+1]-start[mu];++k){
            int i=start[mu]+k;if(int(m[4*i])!=mu)throw std::runtime_error("Noncontiguous basic moments");
            std::array<int,3> pow{{int(m[4*i+1]),int(m[4*i+2]),int(m[4*i+3])}};
            if(!mu){
                powers[k]=pow;
                if(k){
                    int d=0;while(d<3 && pow[d]==0)++d;if(d==3)throw std::runtime_error("Duplicate constant");
                    --pow[d];auto at=std::find(powers.begin(),powers.begin()+k,pow);
                    if(at==powers.begin()+k)throw std::runtime_error("Monomial order");
                    parent[k]=int(at-powers.begin());axis[k]=d;
                }
            }else if(powers[k]!=pow)throw std::runtime_error("MTP monomial prefix differs");
        }
        auto t=numbers(field("alpha_index_times"));if(t.size()%4)throw std::runtime_error("Contraction list");
        int prev=B;
        for(size_t i=0;i<t.size();i+=4){
            int a=int(t[i]),b=int(t[i+1]),dest=int(t[i+3]);
            if(dest<prev||dest>=M||a<0||b<0||a>=dest||b>=dest)throw std::runtime_error("Contraction ordering");
            ++product_start[dest+1];prev=dest;products.push_back({(unsigned short)a,(unsigned short)b,t[i+2]});
        }
        std::partial_sum(product_start.begin(),product_start.end(),product_start.begin());
        for(double x:numbers(field("alpha_moment_mapping"))){
            if(x<0||x>=M)throw std::runtime_error("Scalar index");
            mapping.push_back(int(x));
        }
        coef=numbers(field("moment_coeffs"));offset=numbers(field("species_coeffs"));
        if(mapping.size()!=coef.size()||offset.size()!=2)throw std::runtime_error("Energy coefficients");
    }
    template<class Getter> void check_dimensions(Getter field){
        if(std::stoi(field("species_count"))!=2||std::stoi(field("radial_basis_size"))!=8||
           std::stoi(field("radial_funcs_count"))!=6||std::stoi(field("alpha_moments_count"))!=M||
           field("radial_basis_type").find("RBChebyshev")!=0)throw std::runtime_error("Unsupported MTP basis");
    }
    void add(Basic& v,const Vec3& dr,int central,int neighbor,double sign) const {
        const double rr=dot(dr,dr);
        if(rr>=rc*rc)return;
        if(rr<=1e-20)throw std::runtime_error("Overlapping atomic positions");
        const double r=std::sqrt(rr), inv=1./r, x=(2*r-rc-rmin)/(rc-rmin);
        const double unit[3]={dr[0]*inv,dr[1]*inv,dr[2]*inv};
        double ch[8];ch[0]=1.;ch[1]=x;for(int i=2;i<8;++i)ch[i]=2*x*ch[i-1]-ch[i-2];
        double mono[K];mono[0]=1.;
        for(int k=1;k<K;++k)mono[k]=mono[parent[k]]*unit[axis[k]];
        const double pref=sign*scaling*(r-rc)*(r-rc);
        for(int mu=0;mu<6;++mu){
            double g=0;for(int n=0;n<8;++n)g+=ch[n]*radial[central][neighbor][mu][n];g*=pref;
            double* out=v.data()+start[mu];
            for(int k=0;k<start[mu+1]-start[mu];++k)out[k]+=g*mono[k];
        }
    }
    void energies(const std::vector<Basic>& basic,const std::vector<int>& types,std::vector<double>& out){
        out.resize(basic.size());
        for(size_t lo=0;lo<basic.size();lo+=L){
            int lanes=std::min<int>(L,int(basic.size()-lo));
            const Basic* ptr[L];for(int k=0;k<L;++k)ptr[k]=&basic[lo+std::min(k,lanes-1)];
            auto* x=contractions.data();
            for(int m=0;m<B;++m)for(int k=0;k<L;++k)x[m][k]=(*ptr[k])[m];
            for(int m=B;m<M;++m){
                Wide z{};
                for(int i=product_start[m];i<product_start[m+1];++i){
                    const auto& p=products[i];const auto& a=x[p.left];const auto& b=x[p.right];
                    for(int k=0;k<L;++k)z[k]+=p.coeff*a[k]*b[k];
                }
                x[m]=z;
            }
            Wide sum{};for(size_t m=0;m<coef.size();++m)for(int k=0;k<L;++k)sum[k]+=coef[m]*x[mapping[m]][k];
            for(int k=0;k<lanes;++k)out[lo+k]=offset.at(types[lo+k])+sum[k];
        }
    }
};
struct Edge{int j; Vec3 shift;};
struct State{
    Cell cell; std::vector<Vec3> pos,reference;
    std::vector<int> types;std::vector<std::vector<Edge>> graph;
    std::vector<Basic> moments;std::vector<double> site_energy;
    double total=0;
};
} // namespace qi_mtp

class Engine {
    using Basic=qi_mtp::Basic;using State=qi_mtp::State;
    static double skin(){return 1.5;}
    qi_mtp::Model model;
    State state;
    std::unique_ptr<State> full_trial;
    std::vector<int> centers,where,species;
    std::vector<Basic> pending;
    std::vector<double> new_energies;
    int changed=-1,new_type=0;
    Vec3 new_pos{}; double delta=0;

    void neighbors(State& s){
        using namespace qi_mtp;
        auto& c=s.cell;double det=dot(c[0],cross(c[1],c[2]));if(!(det>0))throw std::runtime_error("Nonpositive volume");
        Cell reciprocal{{mul(cross(c[1],c[2]),1./det),mul(cross(c[2],c[0]),1./det),mul(cross(c[0],c[1]),1./det)}};
        double radius=model.rc+skin();Vec3 bound;
        for(int d=0;d<3;++d)bound[d]=radius*std::sqrt(dot(reciprocal[d],reciprocal[d]));
        s.reference=s.pos;s.graph.assign(s.pos.size(),{});
        for(int i=0;i<int(s.pos.size());++i)for(int j=i;j<int(s.pos.size());++j){
            Vec3 diff=sub(s.pos[j],s.pos[i]);int low[3],high[3];
            for(int d=0;d<3;++d){
                double q=dot(diff,reciprocal[d]);low[d]=int(std::ceil(-q-bound[d]-1e-12));high[d]=int(std::floor(-q+bound[d]+1e-12));
            }
            for(int a=low[0];a<=high[0];++a)for(int b=low[1];b<=high[1];++b)for(int h=low[2];h<=high[2];++h){
                if(i==j && a==0 && b==0 && h==0)continue;
                Vec3 shift,dr;for(int d=0;d<3;++d){shift[d]=a*c[0][d]+b*c[1][d]+h*c[2][d];dr[d]=diff[d]+shift[d];}
                if(dot(dr,dr)<=radius*radius){
                    s.graph[i].push_back({j,shift});if(i!=j)s.graph[j].push_back({i,mul(shift,-1.)});
                }
            }
        }
    }
    void rebuild(State& s){
        neighbors(s);s.moments.assign(s.pos.size(),{});
        for(int i=0;i<int(s.pos.size());++i)for(auto& edge:s.graph[i]){
            Vec3 r;for(int d=0;d<3;++d)r[d]=s.pos[edge.j][d]-s.pos[i][d]+edge.shift[d];
            model.add(s.moments[i],r,s.types[i],s.types[edge.j],1.);
        }
        model.energies(s.moments,s.types,s.site_energy);
        s.total=std::accumulate(s.site_energy.begin(),s.site_energy.end(),0.);
    }
    void reset_pending(){
        for(int i:centers)where[i]=-1;
        centers.clear();pending.clear();species.clear();new_energies.clear();full_trial.reset();changed=-1;delta=0;
    }
    int include(int j){
        if(where[j]<0){
            where[j]=int(centers.size());centers.push_back(j);
            pending.push_back(state.moments[j]);species.push_back(state.types[j]);
        }
        return where[j];
    }
    void ensure_idle()const{if(changed>=0||full_trial)throw std::logic_error("An energy trial is already pending");}
    std::unique_ptr<State> clone_geometry()const{
        std::unique_ptr<State> p(new State);
        p->cell=state.cell;p->pos=state.pos;p->types=state.types;return p;
    }
public:
    Engine(const std::string& file,Cell cell,std::vector<Vec3> positions,std::vector<int> types):model(file){
        if(positions.size()!=types.size()||types.empty())throw std::invalid_argument("Invalid positions and types");
        for(int t:types)if(t!=0&&t!=1)throw std::invalid_argument("Expected Ti=0, Al=1");
        state.cell=cell;state.pos=std::move(positions);state.types=std::move(types);
        where.assign(state.pos.size(),-1);pending.reserve(96);rebuild(state);
    }
    double energy()const{return state.total;}
    double trial(int site,Vec3 position,int type){
        using namespace qi_mtp;
        ensure_idle();if(site<0||site>=int(state.pos.size())||(type!=0&&type!=1))throw std::invalid_argument("Single-site trial");
        auto far=[&](){Vec3 d=sub(position,state.reference[site]);return dot(d,d)>0.25*skin()*skin();};
        if(far()){
            neighbors(state);
            if(far()){
                full_trial=clone_geometry();full_trial->pos[site]=position;full_trial->types[site]=type;
                rebuild(*full_trial);return delta=full_trial->total-state.total;
            }
        }
        changed=site;new_pos=position;new_type=type;include(site);
        pending[0].fill(0.);species[0]=type;
        auto old_position=state.pos[site];int old_type=state.types[site];double rc2=model.rc*model.rc;
        for(auto& edge:state.graph[site]){
            int j=edge.j;Vec3 old_r,new_r;
            for(int d=0;d<3;++d){
                old_r[d]=state.pos[j][d]-old_position[d]+edge.shift[d];
                new_r[d]=(j==site?position[d]:state.pos[j][d])-position[d]+edge.shift[d];
            }
            if(dot(old_r,old_r)>=rc2 && dot(new_r,new_r)>=rc2)continue;
            model.add(pending[0],new_r,type,j==site?type:state.types[j],1.);
            if(j!=site){
                int k=include(j);
                model.add(pending[k],mul(old_r,-1.),state.types[j],old_type,-1.);
                model.add(pending[k],mul(new_r,-1.),state.types[j],type,1.);
            }
        }
        model.energies(pending,species,new_energies);
        for(size_t k=0;k<centers.size();++k)delta+=new_energies[k]-state.site_energy[centers[k]];
        return delta;
    }
    double set_volume_trial(double scale){
        ensure_idle();if(!(scale>0.)||!std::isfinite(scale))throw std::invalid_argument("Positive finite linear scale required");
        full_trial=clone_geometry();for(auto& r:full_trial->cell)for(double& x:r)x*=scale;
        for(auto& r:full_trial->pos)for(double& x:r)x*=scale;
        rebuild(*full_trial);return delta=full_trial->total-state.total;
    }
    void accept(){
        if(full_trial){state=std::move(*full_trial);reset_pending();return;}
        if(changed<0)throw std::logic_error("No pending trial");
        state.pos[changed]=new_pos;state.types[changed]=new_type;state.total+=delta;
        for(size_t k=0;k<centers.size();++k){state.moments[centers[k]]=pending[k];state.site_energy[centers[k]]=new_energies[k];}
        reset_pending();
    }
    void reject(){if(changed<0&&!full_trial)throw std::logic_error("No pending trial");reset_pending();}
};
