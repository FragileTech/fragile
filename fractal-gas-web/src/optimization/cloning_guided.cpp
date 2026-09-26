#include "optimization/cloning_guided.hpp"
#include "optimization/adaptive.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>

namespace fg::optimization {
namespace {
using Mat=Eigen::MatrixXd;
using Vec=Eigen::VectorXd;
void str(Json& j,const char* key,const std::string& value) {j.object[key].kind=Json::String;j.object[key].string=value;}
}
struct CloningGuided::Impl {
  struct Model { std::vector<float> anchor;Mat shape,factor;Vec drift,field;size_t comparisons=0;double condition=1; };
  int d;double width,low=.0001,high=1,upper=1,strength=.25;
  bool periodic=false,minimize=true,geometry_enabled=true,drift_enabled=true,dirty=false;
  std::string mode="velocity";
  uint64_t version=0;
  size_t fallback=0;
  fractal::FrozenPopulation pending;
  fractal::SelectionEvidence evidence;
  std::vector<double> ranks;
  std::vector<Model> models;
  Impl(const Benchmark& b):d(b.d),width(b.high-b.low) {}
  Vec offset(const float* x,const float* anchor) const {
    Vec y(d);for(int j=0;j<d;++j) {double v=double(x[j])-anchor[j];if(periodic) v=std::remainder(v,width);y[j]=v/width;}return y;
  }
  size_t nearest(const float* x) const {
    size_t best=0;double distance=INFINITY;
    for(size_t i=0;i<models.size();++i) {double v=offset(x,models[i].anchor.data()).squaredNorm();if(v<distance) {distance=v;best=i;}}
    return best;
  }
  Model fresh(const float* x) const {
    Model m;m.anchor.assign(x,x+d);m.drift=Vec::Zero(d);m.field=Vec::Zero(d);
    if(d<=64) {m.shape=Mat::Identity(d,d);m.factor=m.shape;}else m.shape=Mat::Ones(d,1);
    return m;
  }
  double scale(double value) const {
    double p=.5;
    if(std::isfinite(value)&&ranks.size()>1&&ranks.front()!=ranks.back()) {
      double cost=minimize?value:-value;
      p=std::clamp((double(std::lower_bound(ranks.begin(),ranks.end(),cost)-ranks.begin())+
          double(std::upper_bound(ranks.begin(),ranks.end(),cost)-ranks.begin())-1)/(2*(ranks.size()-1)),0.,1.);
    }
    return low>0?std::exp((1-p)*std::log(low)+p*std::log(upper)):p*upper;
  }
  void draw(const float* x,float* delta,double scale,Rng& rng) const {
    Vec z(d);for(int j=0;j<d;++j) z[j]=normal(rng);
    Vec y=z;
    if(!models.empty()) {
      const auto& m=models[nearest(x)];
      if(geometry_enabled) {
        if(d<=64) y=m.factor*z;
        else {y=m.shape.col(0).array().sqrt()*z.array();for(int k=1;k<m.shape.cols();++k) y+=m.shape.col(k)*normal(rng);}
      }
      if(drift_enabled) y+=m.drift;
    }
    for(int j=0;j<d;++j) delta[j]=float(scale*y[j]);
  }
  void fit(Model& m,const std::vector<size_t>& rows) {
    // Selection comparisons describe a density field, not independent trials.
    // Keep their multiplicity; ancestry is intentionally irrelevant here.
    constexpr double radius=.1, shrink=.1;
    Mat x(d,rows.size());Vec mean=Vec::Zero(d),field=Vec::Zero(d);
    std::vector<double> weights;weights.reserve(rows.size());
    double total=0,field_weight=0;size_t count=0;
    m.drift.setZero();m.field.setZero();m.comparisons=0;
    for(size_t i:rows) {
      const size_t donor=size_t(evidence.donors[i]);
      const float* origin=pending.positions.data()+i*d;
      Vec direction=offset(pending.positions.data()+donor*d,origin);
      const double rms=direction.norm()/std::sqrt(double(d));
      if(rms==0) continue;
      direction*=std::min(1.,radius/rms);
      const double response=std::tanh(evidence.score[i]);
      const double weight=std::exp(-offset(origin,m.anchor.data()).squaredNorm()/(2*d*radius*radius));
      Vec selection=response*direction;
      x.col(count++)=selection;weights.push_back(weight);
      mean+=weight*selection;total+=weight;
      field+=weight*selection;field_weight+=weight*std::abs(response);
    }
    m.comparisons=count;
    if(field_weight>0) m.field=field/(radius*field_weight);
    m.drift=strength*m.field;
    if(count<2 || total<=0) {++fallback;return;}
    mean/=total;x.conservativeResize(d,count);
    for(size_t k=0;k<count;++k) x.col(k)=std::sqrt(weights[k]/total)*(x.col(k)-mean);
    const double trace=x.squaredNorm();
    // No new orientation evidence: preserve compatible geometry, not identity.
    if(!std::isfinite(trace)||trace<1e-20) {++fallback;return;}
    if(d<=64) {
      Mat shape=(1-shrink)*double(d)/trace*(x*x.transpose());shape.diagonal().array()+=shrink;
      Eigen::SelfAdjointEigenSolver<Mat> eig(shape);
      if(eig.info()!=Eigen::Success) {++fallback;return;}
      Vec values=eig.eigenvalues().cwiseMax(std::max(1e-12,eig.eigenvalues().maxCoeff()/1e6));
      values*=d/values.sum();m.condition=values.maxCoeff()/values.minCoeff();
      m.shape=eig.eigenvectors()*values.asDiagonal()*eig.eigenvectors().transpose();
      Eigen::LLT<Mat> chol(m.shape);if(chol.info()!=Eigen::Success) {++fallback;m=fresh(m.anchor.data());return;}
      m.factor=chol.matrixL();
    } else {
      // Deterministic block power iteration: never allocate a d-by-d matrix.
      const int rank=std::min<int>(8,std::min(d,int(x.cols())));Mat basis(d,rank);
      for(int j=0;j<d;++j) for(int k=0;k<rank;++k) basis(j,k)=std::sin(double((j+1)*(k+1)));
      for(int k=0;k<12;++k) {Mat next=x*(x.transpose()*basis);Eigen::HouseholderQR<Mat> qr(next);basis=qr.householderQ()*Mat::Identity(d,rank);}
      Mat projected=basis.transpose()*x;Eigen::SelfAdjointEigenSolver<Mat> eig(projected*projected.transpose());
      if(eig.info()!=Eigen::Success) {++fallback;return;}
      Mat components=basis*eig.eigenvectors()*eig.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
      Vec residual=x.array().square().rowwise().sum().matrix()-components.array().square().rowwise().sum().matrix();
      m.shape.resize(d,rank+1);m.shape.col(0)=((1-shrink)*d/trace*residual.cwiseMax(0).array()+shrink).matrix();
      m.shape.rightCols(rank)=std::sqrt((1-shrink)*d/trace)*components;
      double bound=m.shape.col(0).maxCoeff()+m.shape.rightCols(rank).squaredNorm();
      m.shape.col(0)=m.shape.col(0).cwiseMax(std::max(1e-12,bound/1e6));
      double norm=d/(m.shape.col(0).sum()+m.shape.rightCols(rank).squaredNorm());
      m.shape.col(0)*=norm;m.shape.rightCols(rank)*=std::sqrt(norm);
      m.condition=(m.shape.col(0).maxCoeff()+m.shape.rightCols(rank).squaredNorm())/m.shape.col(0).minCoeff();
    }

  }
};
CloningGuided::CloningGuided(const Benchmark& b,const Json& c):impl(std::make_unique<Impl>(b)) {configure(c);}
CloningGuided::CloningGuided(const CloningGuided& other):impl(std::make_unique<Impl>(*other.impl)) {}
CloningGuided::~CloningGuided()=default;
void CloningGuided::configure(const Json& c) {
  auto& s=*impl;
  double low=bounded(c["adaptive_min_scale"],.0001,0,1e6,"minimum movement scale");
  double high=bounded(c["adaptive_max_scale"],1,low,1e6,"maximum movement scale");
  double fraction=bounded(c["adaptive_round_fraction"],1,0,1,"round scale fraction");
  bool periodic=c["periodic"].flag(false);auto mode=c["adaptive_euclidean_mode"].str("velocity");
  if(mode!="velocity"&&mode!="position") throw std::invalid_argument("Invalid Euclidean movement mode");
  if(periodic!=s.periodic||mode!=s.mode) reset();
  s.periodic=periodic;s.mode=mode;s.minimize=c["objective"].str("minimize")=="minimize";
  s.low=low;s.high=high;s.upper=low>0?std::exp((1-fraction)*std::log(low)+fraction*std::log(high)):fraction*high;
  s.strength=bounded(c["cloning_drift_strength"],.25,0,1,"cloning drift strength");
  s.geometry_enabled=c["cloning_geometry"].flag(true);s.drift_enabled=c["cloning_drift"].flag(true);
  for(auto& m:s.models) m.drift=s.strength*m.field;
}
void CloningGuided::freeze_population(const fractal::FrozenPopulation& p) {
  auto& s=*impl;s.ranks.clear();
  for(size_t i=0;i<p.objectives.size();++i) if(p.valid[i]&&std::isfinite(p.objectives[i])) s.ranks.push_back(s.minimize?p.objectives[i]:-p.objectives[i]);
  std::sort(s.ranks.begin(),s.ranks.end());
}
void CloningGuided::observe_cloning(const fractal::FrozenPopulation& p,const fractal::SelectionEvidence& e) {
  const size_t n=p.valid.size();
  if(p.dimensions!=impl->d||p.positions.size()!=n*impl->d||e.score.size()!=n||e.donors.size()!=n||(!e.active.empty()&&e.active.size()!=n)) throw std::invalid_argument("Cloning snapshot shape");
  for(size_t i=0;i<n;++i)
    if(e.donors[i]<0 || size_t(e.donors[i])>=n || std::isnan(e.score[i]))
      throw std::invalid_argument("Invalid cloning comparison");
  impl->pending=p;impl->evidence=e;impl->dirty=true;
}
void CloningGuided::update() {
  auto& s=*impl;if(!s.dirty) return;
  s.fallback=0;
  std::vector<size_t> valid;
  for(size_t i=0;i<s.pending.valid.size();++i) if((s.evidence.active.empty()||s.evidence.active[i])&&s.pending.valid[i]&&s.pending.valid[size_t(s.evidence.donors[i])]) {
    bool finite=true;for(int j=0;j<s.d;++j) finite &= std::isfinite(s.pending.positions[i*s.d+j]) && std::isfinite(s.pending.positions[size_t(s.evidence.donors[i])*s.d+j]);
    if(!finite) continue;
    valid.push_back(i);const float* x=s.pending.positions.data()+i*s.d;
    if(s.models.empty()||(s.models.size()<16 && s.offset(x,s.models[s.nearest(x)].anchor.data()).norm()/std::sqrt(double(s.d))>.1)) s.models.push_back(s.fresh(x));
  }
  std::vector<std::vector<size_t>> rows(s.models.size());
  for(size_t i:valid) rows[s.nearest(s.pending.positions.data()+i*s.d)].push_back(i);
  for(size_t i=0;i<s.models.size();++i) s.fit(s.models[i],rows[i]);
  s.pending={};s.evidence={};s.dirty=false;++s.version;
}
void CloningGuided::reset() {impl->models.clear();impl->pending={};impl->evidence={};impl->dirty=false;++impl->version;}
void CloningGuided::sample(const float* x,float* delta,int d,Rng& rng) const {
  if(d!=impl->d) throw std::invalid_argument("Proposal dimensions");
  impl->draw(x,delta,impl->scale(NAN),rng);
}
fractal::ProposalTrial CloningGuided::propose_with_objective(const float* x,float* delta,int d,Rng& rng,fractal::TrialIdentity id,double objective) const {
  if(d!=impl->d) throw std::invalid_argument("Proposal dimensions");
  fractal::ProposalTrial trial;id.model=impl->version;trial.identity=id;trial.origin.assign(x,x+d);trial.scale=impl->scale(objective);
  impl->draw(x,delta,trial.scale,rng);trial.direction.resize(d);
  for(int j=0;j<d;++j) trial.direction[j]=trial.scale>0?delta[j]/trial.scale:0;
  return trial;
}
void CloningGuided::continue_trial(const fractal::ProposalTrial& trial,float* delta,int d,Rng& rng) const {
  if(d!=impl->d) throw std::invalid_argument("Proposal dimensions");
  impl->draw(trial.origin.data(),delta,trial.scale,rng);
}
Json CloningGuided::diagnostics() const {
  auto& s=*impl;Json j;j.kind=Json::Object;double comparisons=0,condition=1,ratio=0;
  for(auto& m:s.models) {comparisons+=m.comparisons;condition=std::max(condition,m.condition);ratio=std::max(ratio,m.drift.norm()/std::sqrt(double(s.d)));}
  j.object["model_count"]=number(s.models.size());j.object["model_version"]=number(s.version);
  j.object["selection_comparisons"]=number(s.models.empty()?0:comparisons/s.models.size());
  j.object["condition_number"]=number(condition);j.object["drift_noise_ratio"]=number(ratio);j.object["fallback_count"]=number(s.fallback);
  j.object["scale_min"]=number(s.low);j.object["scale_max"]=number(s.upper);
  str(j,"weighting","signed_clone_score");str(j,"strategy","cloning_guided");
  str(j,"fallback_reason",s.fallback?"No directional variance or failed factorization":"none");return j;
}
double CloningGuided::visual_scale(double objective) const {return impl->scale(objective);}
Json CloningGuided::visual_geometry() const {
  Json list=geometry();
  for(size_t i=0;i<list.array.size();++i) {
    auto& j=list.array[i];const auto& m=impl->models[i];
    str(j,"representation",impl->d<=64?"dense":"diagonal_low_rank");
    j.object["scale"]=number(impl->scale(NAN));
    j.object["field"]=array(std::vector<double>(m.field.data(),m.field.data()+impl->d));
    j.object["drift"]=array(std::vector<double>(m.drift.data(),m.drift.data()+impl->d));
    if(!impl->drift_enabled) j.object["drift"]=array(std::vector<double>(impl->d,0));
    j.object["drift_enabled"].kind=Json::Boolean;j.object["drift_enabled"].number=impl->drift_enabled;
    if(!impl->geometry_enabled) {
      str(j,"representation","diagonal");j.object["columns"]=number(1);
      j.object["shape"]=array(std::vector<double>(impl->d,1));
    }
  }
  return list;
}
Json CloningGuided::geometry() const {
  Json list;list.kind=Json::Array;auto& s=*impl;
  for(const auto& m:s.models) {
    Json j;j.kind=Json::Object;str(j,"strategy","cloning_guided");str(j,"mode",s.mode);
    j.object["version"]=number(1);j.object["dimensions"]=number(s.d);j.object["columns"]=number(m.shape.cols());j.object["scale"]=number(1);
    j.object["anchor"]=array(std::vector<double>(m.anchor.begin(),m.anchor.end()));
    std::vector<double> values;for(int i=0;i<s.d;++i) for(int k=0;k<m.shape.cols();++k) values.push_back(m.shape(i,k));j.object["shape"]=array(values);list.array.push_back(j);
  }return list;
}
void CloningGuided::restore_geometry(const Json& list) {
  if(list.items().size()>64) throw std::invalid_argument("Too many movement models");
  std::vector<Impl::Model> models;auto& s=*impl;
  for(const auto& j:list.items()) {
    if(j["strategy"].str()!="cloning_guided") continue;
    validate_adaptive_geometry(j,s.d);if(j["mode"].str()!=s.mode) continue;
    std::vector<float> anchor;for(const auto& v:j["anchor"].items()) anchor.push_back(float(v.num()));auto m=s.fresh(anchor.data());
    int cols=int(j["columns"].num());m.shape.resize(s.d,cols);
    for(int i=0;i<s.d;++i) for(int k=0;k<cols;++k) m.shape(i,k)=j["shape"].array[size_t(i)*cols+k].num();
    if(s.d<=64) {Eigen::LLT<Mat> chol(m.shape);m.factor=chol.matrixL();}
    models.push_back(m);if(models.size()==16) break;
  }s.models=std::move(models);s.pending={};s.evidence={};s.dirty=false;++s.version;
}
}
