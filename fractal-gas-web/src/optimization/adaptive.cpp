#include "optimization/adaptive.hpp"
#include <Eigen/Dense>
#include <array>
#include <deque>
#include <set>

namespace fg::optimization {
using namespace fractal;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
struct AdaptiveExploration::Impl {
  struct Model {
    std::vector<float> anchor;
    Matrix shape, factor;
    Vector path;
    uint64_t touched = 0, last_update = 0;
    bool has_update = false;
    std::deque<TrialOutcome> outcomes;
  };
  int d;
  double width, sigma, minimum_scale=.0001, maximum_scale=1, round_maximum=1;
  bool periodic, active, paths, scales, differences, pairs, mixture, minimize=true;
  std::vector<double> ranked_objectives;
  uint64_t version = 0, clock = 0, paired = 0, observations = 0;
  double effective_parents = 0;
  std::array<double, 3> probabilities{.7, .15, .15}, rates{.5, .5, .5};
  FrozenPopulation population;
  std::vector<Model> models;
  std::deque<Model> retired;
  std::string mode="velocity";
  std::deque<TrialOutcome> pending;
  std::array<double,3> pending_wins{},pending_costs{};
  // A bounded identity cache complements each model's bounded learning window.
  std::set<TrialIdentity> seen;
  std::deque<TrialIdentity> order;
  Impl(const Benchmark& b) : d(b.d), width(b.high-b.low), sigma(1), periodic(false) {}
  static double interpolate(double low, double high, double fraction) {
    if (fraction<=0 || low==high) return low;
    if (fraction>=1) return high;
    return low>0 ? std::exp((1-fraction)*std::log(low)+fraction*std::log(high))
                 : fraction*high;
  }
  double percentile(double objective) const {
    const auto& values=ranked_objectives;
    if(!std::isfinite(objective) || values.size()<2 || values.front()==values.back()) return .5;
    const double cost=minimize?objective:-objective;
    const auto lo=std::lower_bound(values.begin(),values.end(),cost);
    const auto hi=std::upper_bound(values.begin(),values.end(),cost);
    // Midranks give tied walkers identical scales. Unseen planner descendants
    // use the same frozen empirical distribution throughout search and replay.
    return std::clamp((double(lo-values.begin())+double(hi-values.begin())-1)/
                      (2*double(values.size()-1)),0.,1.);
  }
  double scale_for(double objective) const {
    return scales ? interpolate(minimum_scale,round_maximum,percentile(objective))
                  : std::clamp(sigma,minimum_scale,round_maximum);
  }
  double distance(const float* a, const float* b) const {
    double sum = 0;
    for (int j=0; j<d; ++j) {
      double v = double(a[j])-b[j];
      if (periodic) v = std::remainder(v, width);
      sum += (v/width)*(v/width);
    }
    return std::sqrt(sum/d);
  }
  size_t nearest(const float* x) const {
    size_t index = 0;
    for (size_t i=1; i<models.size(); ++i)
      if (distance(x, models[i].anchor.data()) < distance(x, models[index].anchor.data())) index=i;
    return index;
  }
  Model fresh(const std::vector<float>& x) const {
    Model m; m.anchor=x;
    // The large-dimensional representation stores diagonal variance followed
    // by up to eight correlated components, never a dense d-by-d matrix.
    m.shape = d<=64 ? Matrix::Identity(d,d).eval() : Matrix::Ones(d,1).eval();
    m.factor=m.shape; m.path=Vector::Zero(d);
    return m;
  }
  Vector sample_shape(const Model& m, Rng& rng) const {
    Vector z(d); for(int j=0;j<d;++j) z[j]=normal(rng);
    if(d<=64) return m.factor*z;
    Vector y=m.shape.col(0).array().sqrt()*z.array();
    for(int k=1;k<m.shape.cols();++k) y += m.shape.col(k)*normal(rng);
    return y;
  }
  Vector whiten(const Model& m, const Vector& y) const {
    if(d<=64) return m.factor.triangularView<Eigen::Lower>().solve(y);
    // Woodbury-compatible symmetric whitening of D + U U^T. Work in
    // diagonal-whitened coordinates and solve only the rank-eight subspace.
    Vector inv=m.shape.col(0).array().sqrt().inverse();
    Vector z=inv.array()*y.array();
    if(m.shape.cols()>1) {
      Matrix u=inv.asDiagonal()*m.shape.rightCols(m.shape.cols()-1);
      Eigen::JacobiSVD<Matrix> svd(u,Eigen::ComputeThinU|Eigen::ComputeThinV);
      Vector correction=(1+svd.singularValues().array().square()).sqrt().inverse()-1;
      z += svd.matrixU()*correction.asDiagonal()*(svd.matrixU().transpose()*z);
    }
    return z;
  }
  void condition(Model& m, const Matrix& positive, const Matrix& negative,
                 const Vector& mean, double neff) {
    const bool pos=positive.cols()>0, neg=negative.cols()>0 && active;
    const double retain=1-(pos?.10:0)-(paths&&pos?.05:0)+(neg?.05:0);
    if(paths&&pos) {
      m.path=.8*m.path+std::sqrt(.2*1.8*neff)*mean;
    }
    if(d<=64) {
      Matrix c=retain*m.shape;
      if(pos) c.noalias()+=.10*positive*positive.transpose();
      if(paths&&pos) c.noalias()+=.05*m.path*m.path.transpose();
      if(neg) c.noalias()-=.05*negative*negative.transpose();
      c=(.5*(c+c.transpose())).eval();
      Eigen::SelfAdjointEigenSolver<Matrix> eigen(c);
      if(eigen.info()!=Eigen::Success || !eigen.eigenvalues().allFinite()) { m.shape=Matrix::Identity(d,d); m.factor=m.shape; return; }
      Vector values=eigen.eigenvalues().cwiseMax(1e-8);
      values=values.cwiseMax(values.maxCoeff()/1e4);
      values*=d/values.sum(); values=.95*values.array()+.05;
      m.shape=eigen.eigenvectors()*values.asDiagonal()*eigen.eigenvectors().transpose();
      Eigen::LLT<Matrix> chol(m.shape);
      if(chol.info()==Eigen::Success) m.factor=chol.matrixL();
      else { m.shape=Matrix::Identity(d,d); m.factor=m.shape; }
    } else {
      Vector diag=retain*m.shape.col(0);
      const int oldrank=int(m.shape.cols())-1;
      Matrix components(d,oldrank+positive.cols()+(paths&&pos?1:0));
      if(oldrank) components.leftCols(oldrank)=std::sqrt(retain)*m.shape.rightCols(oldrank);
      if(pos) components.middleCols(oldrank,positive.cols())=std::sqrt(.10)*positive;
      if(paths&&pos) components.col(components.cols()-1)=std::sqrt(.05)*m.path;
      if(neg) diag-=.05*negative.array().square().rowwise().sum().matrix();
      int rank=0;
      Matrix kept(d,0);
      const int negative_columns=neg?int(negative.cols()):0;
      Matrix combined(d,components.cols()+negative_columns);
      combined.leftCols(components.cols())=components;
      if(negative_columns) combined.rightCols(negative_columns)=negative;
      if(combined.cols()>0) {
        // Compress the signed update in its small evidence subspace. Negative
        // evidence must affect the retained correlation directions as well as
        // the diagonal; never form a dense d-by-d covariance here.
        Eigen::HouseholderQR<Matrix> qr(combined);
        Matrix basis=qr.householderQ()*Matrix::Identity(d,std::min(d,int(combined.cols())));
        Matrix positive_projection=basis.transpose()*components;
        Matrix reduced=positive_projection*positive_projection.transpose();
        if(negative_columns) {
          Matrix negative_projection=basis.transpose()*negative;
          reduced.noalias()-=.05*negative_projection*negative_projection.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Matrix> eigen(reduced);
        if(eigen.info()==Eigen::Success && eigen.eigenvalues().allFinite()) {
          for(int i=int(eigen.eigenvalues().size())-1;i>=0 && rank<8 && eigen.eigenvalues()[i]>1e-12;--i) ++rank;
          if(rank) kept=basis*eigen.eigenvectors().rightCols(rank)*eigen.eigenvalues().tail(rank).array().sqrt().matrix().asDiagonal();
        }
      }
      diag+=components.array().square().rowwise().sum().matrix()-kept.array().square().rowwise().sum().matrix();
      double upper=diag.maxCoeff()+kept.squaredNorm();
      diag=diag.cwiseMax(std::max(1e-8,upper/1e4));
      double norm=d/(diag.sum()+kept.squaredNorm());
      m.shape.resize(d,rank+1); m.shape.col(0)=(.95*norm*diag.array()+.05).matrix();
      if(rank) m.shape.rightCols(rank)=std::sqrt(.95*norm)*kept;
    }
  }
};
AdaptiveExploration::AdaptiveExploration(const Benchmark& b,const Json& c) : impl(std::make_unique<Impl>(b)) { configure(c); }
AdaptiveExploration::AdaptiveExploration(const AdaptiveExploration& o) : impl(std::make_unique<Impl>(*o.impl)) {}
AdaptiveExploration::~AdaptiveExploration()=default;
void AdaptiveExploration::configure(const Json& c) {
  auto& s=*impl;
  const uint64_t coordinates=uint64_t(s.d);
  const uint64_t shape=coordinates*(s.d<=64?coordinates:9)*sizeof(double)*2;
  const uint64_t evidence=4096*(coordinates*28+512);
  if(32*shape+evidence>128*1024*1024)
    throw std::invalid_argument("Adaptive model exceeds 128 MiB state budget");
  const bool wrapping=c["periodic"].flag(false);
  const double sigma=bounded(c["perturbation_std"],1,0,1e6,"perturbation standard deviation");
  const double minimum=bounded(c["adaptive_min_scale"],.0001,0,1e6,"minimum movement scale");
  const double maximum=bounded(c["adaptive_max_scale"],1,minimum,1e6,"maximum movement scale");
  const double fraction=bounded(c["adaptive_round_fraction"],1,0,1,"round scale fraction");
  const bool active=c["adaptive_active"].flag(true), paths=c["adaptive_paths"].flag(true),
      scales=c["adaptive_scale"].flag(true), differences=c["adaptive_difference"].flag(true),
      pairs=c["adaptive_pairs"].flag(true), mixture=c["adaptive_mixture"].flag(true);
  // Validate the complete configuration before changing learned state.
  if(wrapping!=s.periodic) reset();
  s.mode=c["adaptive_euclidean_mode"].str("velocity");
  s.periodic=wrapping; s.sigma=sigma; s.active=active; s.paths=paths;
  s.minimum_scale=minimum;s.maximum_scale=maximum;
  s.round_maximum=Impl::interpolate(minimum,maximum,fraction);
  s.minimize=c["objective"].str("minimize")=="minimize";
  s.scales=scales; s.differences=differences; s.pairs=pairs; s.mixture=mixture;
  if(!s.differences) { double sum=s.probabilities[0]+s.probabilities[1]; s.probabilities={s.probabilities[0]/sum,s.probabilities[1]/sum,0}; }
  else if(s.probabilities[2]==0) s.probabilities={.7,.15,.15};
}
void AdaptiveExploration::reset() {
  auto& s=*impl; s.models.clear(); s.retired.clear(); s.pending.clear(); s.pending_wins={};s.pending_costs={}; s.seen.clear(); s.order.clear(); s.population={};
  s.probabilities={.7,.15,.15}; s.rates={.5,.5,.5}; s.effective_parents=0; ++s.version;
  s.ranked_objectives.clear();
}
void AdaptiveExploration::freeze_population(const FrozenPopulation& p) {
  if(p.dimensions!=impl->d || p.positions.size()!=p.families.size()*p.dimensions || p.valid.size()!=p.families.size() ||
     (!p.objectives.empty() && p.objectives.size()!=p.families.size()))
    throw std::invalid_argument("Invalid frozen proposal population");
  for(size_t i=0;i<p.valid.size();++i) if(p.valid[i])
    for(int j=0;j<p.dimensions;++j)
      if(!std::isfinite(p.positions[i*p.dimensions+j]))
        throw std::invalid_argument("Valid donor must have finite coordinates");
  impl->population=p;
  auto& ranked=impl->ranked_objectives;ranked.clear();
  for(size_t i=0;i<p.objectives.size();++i) if(p.valid[i] && std::isfinite(p.objectives[i]))
    ranked.push_back(impl->minimize?p.objectives[i]:-p.objectives[i]);
  std::sort(ranked.begin(),ranked.end());
}
ProposalTrial AdaptiveExploration::propose(const float* x,float* delta,int d,Rng& rng,TrialIdentity id) const {
  return propose_with_objective(x,delta,d,rng,id,NAN);
}
ProposalTrial AdaptiveExploration::propose_with_objective(const float* x,float* delta,int d,Rng& rng,TrialIdentity id,double objective) const {
  const auto& s=*impl;
  if(d!=s.d) throw std::invalid_argument("Proposal dimensionality mismatch");
  ProposalTrial trial; trial.identity=id; trial.identity.model=s.version;
  trial.origin.assign(x,x+d); trial.scale=s.scale_for(objective);
  const double choice=rng.uniform01();
  int family=choice<s.probabilities[0]?0:choice<s.probabilities[0]+s.probabilities[1]?1:2;
  Vector y(d);
  if(family==2) {
    std::vector<size_t> donors;
    for(size_t i=0;i<s.population.families.size();++i)
      if(s.population.valid[i] && s.population.families[i]!=id.parent) donors.push_back(i);
    if(donors.size()>1) {
      size_t a=donors[size_t(rng.randint(0,donors.size()))];
      donors.erase(std::remove_if(donors.begin(),donors.end(),[&](size_t i){return s.population.families[i]==s.population.families[a];}),donors.end());
      if(!donors.empty()) {
        size_t b=donors[size_t(rng.randint(0,donors.size()))];
        for(int j=0;j<d;++j) {
          double diff=double(s.population.positions[a*d+j])-s.population.positions[b*d+j];
          if(s.periodic) diff=std::remainder(diff,s.width);
          y[j]=diff;
        }
        const double rms=y.norm()/std::sqrt(double(d));
        if(rms>0 && std::isfinite(rms)) {
          // Donor separation supplies direction, not an unbounded step size.
          y*=std::sqrt(double(d))/y.norm();
          for(int j=0;j<d;++j) y[j]+=.01*normal(rng);
          y*=trial.scale*std::sqrt(double(d))/y.norm();
        } else family=1;
      } else family=1;
    } else family=1;
  }
  if(family==0 && !s.models.empty()) {
    const auto& model=s.models[s.nearest(x)];
    y=trial.scale*s.sample_shape(model,rng);
  } else if(family!=2) {
    if(family==1) trial.scale=std::min(s.round_maximum,4*trial.scale);
    for(int j=0;j<d;++j) y[j]=trial.scale*normal(rng);
  }
  trial.family=ProposalFamily(family); trial.paired=s.pairs && rng.uniform01()<.1;
  trial.direction.resize(d);
  for(int j=0;j<d;++j) { delta[j]=float(y[j]); trial.direction[j]=trial.scale>0?y[j]/trial.scale:0; }
  return trial;
}
void AdaptiveExploration::sample(const float* x,float* delta,int d,Rng& rng) const { propose(x,delta,d,rng,{}); }
void AdaptiveExploration::observe_external(TrialOutcome o) {
  // Diagnostic adapter has its own model clock; source identity/parent and
  // measured direction are retained. It does not select proposal families.
  o.trial.identity.model=impl->version;
  feedback(o);
}
double AdaptiveExploration::visual_scale(double objective) const {return impl->scale_for(objective);}
Json AdaptiveExploration::visual_geometry() const {
  Json list=geometry();
  list.array.resize(impl->models.size()); // retired archive models are not active
  for(auto& j:list.array) {
    j.object["representation"].kind=Json::String;
    j.object["representation"].string=impl->d<=64?"dense":"diagonal_low_rank";
    j.object["scale"]=number(impl->scale_for(NAN));
  }
  return list;
}
void AdaptiveExploration::feedback(const TrialOutcome& o) {
  auto& s=*impl;
  if(o.replay || o.trial.identity.model!=s.version || o.trial.origin.size()!=size_t(s.d) || o.displacement.size()!=size_t(s.d) || o.trial.direction.size()!=size_t(s.d) || o.evaluations==0) return;
  if(!std::isfinite(o.improvement) || !std::isfinite(o.standard_error) || o.standard_error<0) return;
  for(int j=0;j<s.d;++j) if(!std::isfinite(o.trial.origin[j]) || !std::isfinite(o.displacement[j]) || !std::isfinite(o.trial.direction[j])) return;
  const int family=int(o.trial.family);
  if(family<0 || family>2) throw std::invalid_argument("Invalid proposal family");
  if(!s.seen.insert(o.trial.identity).second) return;
  s.order.push_back(o.trial.identity);
  if(s.order.size()>8192) { s.seen.erase(s.order.front()); s.order.pop_front(); }
  s.pending_costs[family]+=o.evaluations;
  if(o.valid && o.improvement>2*o.standard_error) s.pending_wins[family]+=1;
  ++s.observations;s.paired+=o.trial.paired;
  s.pending.push_back(o);if(s.pending.size()>2048) s.pending.pop_front();
}
void AdaptiveExploration::update() {
  auto& s=*impl;
  if(s.pending.empty()) return;
  const auto wins=s.pending_wins,costs=s.pending_costs;
  s.pending_wins={};s.pending_costs={};
  std::set<size_t> changed;
  for(const auto& o:s.pending) {
    if(!o.valid) continue;
    size_t index=s.models.empty()?0:s.nearest(o.trial.origin.data());
    if(s.models.empty() || s.distance(o.trial.origin.data(),s.models[index].anchor.data())>.1) {
      if(s.models.size()<16) { index=s.models.size(); s.models.push_back(s.fresh(o.trial.origin)); }
      else { index=size_t(std::min_element(s.models.begin(),s.models.end(),[](const auto& a,const auto& b){return a.touched<b.touched;})-s.models.begin()); auto retired=s.models[index];retired.outcomes.clear();
        s.retired.push_back(std::move(retired));if(s.retired.size()>16) s.retired.pop_front();
        s.models[index]=s.fresh(o.trial.origin); }
    }
    auto& m=s.models[index]; m.touched=++s.clock; m.outcomes.push_back(o);
    if(m.outcomes.size()>128) m.outcomes.pop_front();
    changed.insert(index);
  }
  s.pending.clear();
  for(size_t index:changed) {
    auto& m=s.models[index];
    std::map<uint64_t,const TrialOutcome*> parents;
    // One observation per independent parent family caps its total weight.
    for(const auto& o:m.outcomes) {
      if(m.has_update && o.trial.identity.model<=m.last_update) continue;
      auto it=parents.find(o.trial.identity.parent);
      if(it==parents.end() || std::abs(o.improvement)>std::abs(it->second->improvement)) parents[o.trial.identity.parent]=&o;
    }
    if(parents.size()<4) continue;
    std::vector<const TrialOutcome*> pos,neg;
    for(const auto& entry:parents) {
      const auto* o=entry.second;
      if(o->improvement>2*o->standard_error) pos.push_back(o);
      else if(o->improvement < -2*o->standard_error && o->trial.family!=ProposalFamily::Difference) neg.push_back(o);
    }
    auto order=[](const auto* a,const auto* b){return a->improvement>b->improvement;};
    std::stable_sort(pos.begin(),pos.end(),order);
    std::stable_sort(neg.begin(),neg.end(),[&](auto* a,auto* b){return order(b,a);});
    Vector mean=Vector::Zero(s.d); double squared=0;
    auto moment=[&](const std::vector<const TrialOutcome*>& group,bool negative) {
      Matrix result(s.d,group.size()); double sum=0;
      for(size_t k=0;k<group.size();++k) sum+=std::log(group.size()+.5)-std::log(k+1.);
      for(size_t k=0;k<group.size();++k) {
        double w=(std::log(group.size()+.5)-std::log(k+1.))/sum;
        Vector y=Eigen::Map<const Vector>(group[k]->trial.direction.data(),s.d);
        if(negative) y*=std::sqrt(s.d/std::max(1e-12,s.whiten(m,y).squaredNorm()));
        else { mean+=w*y; squared+=w*w; }
        result.col(k)=std::sqrt(w)*y;
      }
      return result;
    };
    Matrix positive=moment(pos,false),negative=moment(neg,true);
    s.effective_parents=squared>0?1/squared:0;
    if(positive.cols() || negative.cols()) s.condition(m,positive,negative,mean,s.effective_parents);
    m.last_update=s.version; m.has_update=true;
  }
  if(s.mixture) {
    double total=0; int enabled=s.differences?3:2;
    for(int f=0;f<enabled;++f) { if(costs[f]>0) s.rates[f]=.8*s.rates[f]+.2*wins[f]/costs[f]; total+=s.rates[f]; }
    for(int f=0;f<enabled;++f) s.probabilities[f]=.1+(1-.1*enabled)*(total>0?s.rates[f]/total:1./enabled);
  }
  ++s.version;
}
void validate_adaptive_geometry(const Json& geometry,int dimensions) {
  if(geometry["version"].num()!=1 || geometry["dimensions"].num()!=dimensions)
    throw std::invalid_argument("Incompatible adaptive geometry");
  const auto& anchor=geometry["anchor"].items();
  const auto& values=geometry["shape"].items();
  const int columns=integer(geometry["columns"],0,dimensions<=64?dimensions:1,dimensions<=64?dimensions:9,"geometry columns");
  if(anchor.size()!=size_t(dimensions) || values.size()!=size_t(dimensions)*columns)
    throw std::invalid_argument("Invalid adaptive geometry shape");
  for(const auto& v:anchor) bounded(v,0,-1e6,1e6,"geometry anchor");
  Matrix shape(dimensions,columns);
  for(int i=0;i<dimensions;++i) for(int j=0;j<columns;++j)
    shape(i,j)=bounded(values[size_t(i)*columns+j],0,-1e6,1e6,"geometry coefficient");
  bounded(geometry["scale"],1,.01,100,"geometry scale");
  const auto mode=geometry["mode"].str();
  if(mode!="velocity" && mode!="position") throw std::invalid_argument("Invalid geometry movement mode");
  if(dimensions<=64) {
    if(!shape.isApprox(shape.transpose(),1e-8)) throw std::invalid_argument("Geometry must be symmetric");
    Eigen::SelfAdjointEigenSolver<Matrix> eigen(shape);
    if(eigen.info()!=Eigen::Success || eigen.eigenvalues().minCoeff()<=0 ||
       eigen.eigenvalues().maxCoeff()/eigen.eigenvalues().minCoeff()>(geometry["strategy"].str()=="cloning_guided"?1000001:10001) ||
       std::abs(shape.trace()-dimensions)>1e-5*dimensions) throw std::invalid_argument("Unsafe covariance geometry");
  } else {
    const double trace=shape.col(0).sum()+shape.rightCols(columns-1).squaredNorm();
    if(shape.col(0).minCoeff()<=0 || std::abs(trace-dimensions)>1e-5*dimensions)
      throw std::invalid_argument("Unsafe low-rank covariance geometry");
  }
}
Json AdaptiveExploration::geometry() const {
  const auto& s=*impl;Json list;list.kind=Json::Array;
  auto append=[&](const auto& model) {
    Json item;item.kind=Json::Object;
    item.object["version"]=number(1);item.object["dimensions"]=number(s.d);
    // Keep the version-one field for archive compatibility. Geometry never
    // restores the obsolete accumulated scalar multiplier.
    item.object["columns"]=number(model.shape.cols());item.object["scale"]=number(1);
    item.object["mode"].kind=Json::String;item.object["mode"].string=s.mode;
    item.object["anchor"].kind=Json::Array;item.object["shape"].kind=Json::Array;
    for(float v:model.anchor) item.object["anchor"].array.push_back(number(v));
    for(int i=0;i<s.d;++i) for(int j=0;j<model.shape.cols();++j) item.object["shape"].array.push_back(number(model.shape(i,j)));
    list.array.push_back(std::move(item));
  };
  for(const auto& model:s.models) append(model);
  for(const auto& model:s.retired) append(model);
  return list;
}
void AdaptiveExploration::restore_geometry(const Json& input) {
  auto& s=*impl;std::vector<Impl::Model> models;
  if(input.items().size()>64) throw std::invalid_argument("Too many warm-start models");
  for(const auto& item:input.items()) {
    if(!item["strategy"].str().empty() && item["strategy"].str()!="adaptive_fractal") continue;
    validate_adaptive_geometry(item,s.d);
    if(item["mode"].str()!=s.mode) continue;
    std::vector<float> anchor;for(const auto& value:item["anchor"].array) anchor.push_back(float(value.num()));
    if(models.size()==16) break;
    auto model=s.fresh(anchor);
    model.shape.resize(s.d,int(item["columns"].num()));
    for(int i=0;i<s.d;++i) for(int j=0;j<model.shape.cols();++j) model.shape(i,j)=item["shape"].array[size_t(i)*model.shape.cols()+j].num();
    if(s.d<=64) {Eigen::LLT<Matrix> factor(model.shape);model.factor=factor.matrixL();}
    models.push_back(std::move(model));
  }
  s.models=std::move(models);s.pending.clear();s.pending_costs={};s.pending_wins={};++s.version;
}
Json AdaptiveExploration::diagnostics() const {
  const auto& s=*impl; Json j; j.kind=Json::Object;
  j.object["model_count"]=number(s.models.size()); j.object["model_version"]=number(s.version);
  j.object["effective_parents"]=number(s.effective_parents);
  j.object["observations"]=number(s.observations); j.object["paired_outcomes"]=number(s.paired);
  double low=s.scale_for(NAN),high=low;
  for(double cost:s.ranked_objectives) {const double value=s.scale_for(s.minimize?cost:-cost);low=std::min(low,value);high=std::max(high,value);}
  j.object["scale_min"]=number(low);j.object["scale_max"]=number(high);
  j.object["scale_limit_min"]=number(s.minimum_scale);j.object["scale_limit_max"]=number(s.maximum_scale);
  j.object["round_scale_max"]=number(s.round_maximum);
  j.object["proposal_probabilities"]=array({s.probabilities[0],s.probabilities[1],s.probabilities[2]});
  return j;
}

EvaluatedProposal evaluate_adaptive_trial(Perturbation& model,const Benchmark& b,const Json& c,
    const float* x,double old,Rng& rng,TrialIdentity identity,int draws,const MovementKernel& move,bool replay) {
  struct Estimate { double mean=0, variance=0; bool valid=true; };
  const int samples=b.stochastic?3:1;
  auto evaluate=[&](const float* point) {
    Estimate estimate;
    for(int k=0;k<samples;++k) {
      double value=b.evaluate_optimization(point,&rng);
      if(!std::isfinite(value)) { estimate.valid=false;continue; }
      double delta=value-estimate.mean;estimate.mean+=delta/(k+1);estimate.variance+=delta*(value-estimate.mean);
    }
    estimate.valid &= b.valid(point);
    estimate.variance=samples>1?estimate.variance/(samples*(samples-1)):0;
    return estimate;
  };
  Estimate before;
  before.mean=old;before.valid=std::isfinite(old);
  const bool baseline=b.stochastic || !before.valid;
  if(baseline) before=evaluate(x);
  std::vector<float> delta(b.d);
  auto trial=model.propose_with_objective(x,delta.data(),b.d,rng,identity,before.mean);
  std::vector<float> noise=delta;
  const auto selected_trial=trial;
  for(int k=1;k<draws;++k) {
    model.continue_trial(selected_trial,delta.data(),b.d,rng);
    noise.insert(noise.end(),delta.begin(),delta.end());
    for(int j=0;j<b.d;++j) trial.direction[j]+=trial.scale>0?delta[j]/trial.scale:0;
  }
  for(auto& direction:trial.direction) direction/=std::sqrt(double(draws));
  const bool minimize=c["objective"].str("minimize")=="minimize";
  EvaluatedProposal selected;selected.position.assign(x,x+b.d);
  double best=minimize?INFINITY:-INFINITY;
  for(int branch=0;branch<(trial.paired?2:1);++branch) {
    const uint64_t evaluation_start=b.evaluations;
    auto candidate=move(noise,branch);
    auto& point=candidate.position;
    if(point.size()!=size_t(b.d)) throw std::logic_error("Movement kernel returned an invalid state");
    const bool outside=!b.valid(point.data());
    b.boundary(point.data(), c["boundary"].str(c["periodic"].flag(false)?"periodic":"none"));
    auto after=evaluate(point.data());
    for(float v:candidate.velocity) after.valid &= std::isfinite(v);
    TrialOutcome outcome;outcome.draws=draws;outcome.trial=trial;outcome.trial.identity.branch=branch;
    outcome.before=before.mean;outcome.after=after.mean;
    outcome.improvement=(minimize?1:-1)*(before.mean-after.mean);
    outcome.standard_error=std::sqrt(before.variance+after.variance);
    outcome.valid=before.valid&&after.valid;outcome.boundary_handled=outside;outcome.replay=replay;
    outcome.evaluations=b.evaluations-evaluation_start+(baseline && branch==0?samples:0);
    outcome.displacement.resize(b.d);
    for(int j=0;j<b.d;++j) {
      outcome.displacement[j]=double(point[j])-x[j];
      if(c["periodic"].flag(false)) outcome.displacement[j]=std::remainder(outcome.displacement[j],b.high-b.low);
      if(branch) outcome.trial.direction[j]*=-1;
    }
    model.observed_feedback(outcome);
    if((after.valid&&!selected.valid) || (after.valid&& (minimize?after.mean<best:after.mean>best)) || branch==0) {
      selected=std::move(candidate);selected.branch=branch;selected.objective=after.mean;selected.valid=after.valid;best=after.mean;
    }
  }
  if(model.observer && selected.valid)
    model.observer->movement(x,selected.position.data(),b.d,identity.parent,replay?"execution":"proposal");
  return selected;
}
uint64_t adaptive_evaluation_bound(const Benchmark& b,const Json& c,bool baseline) {
  return (b.stochastic?3:1)*uint64_t(1+(c["perturbation"].str()!="cloning_guided" && c["adaptive_pairs"].flag(true))+(b.stochastic||baseline));
}
EvaluatedProposal evaluate_adaptive_position(Perturbation& model,const Benchmark& b,const Json& c,
    const float* x,double old,Rng& rng,TrialIdentity identity,bool replay,int draws) {
  return evaluate_adaptive_trial(model,b,c,x,old,rng,identity,draws,
    [&](const std::vector<float>& noise,int branch) {
      EvaluatedProposal result;result.position.assign(x,x+b.d);
      for(int k=0;k<draws;++k) for(int j=0;j<b.d;++j)
        result.position[j]+=(branch?-1:1)*noise[k*b.d+j];
      return result;
    },replay);
}
void AdaptiveExploration::continue_trial(const ProposalTrial& trial,float* delta,int d,Rng& rng) const {
  const auto& s=*impl;
  if(d!=s.d) throw std::invalid_argument("Proposal dimensionality mismatch");
  Vector y(d);
  if(trial.family==ProposalFamily::Local && !s.models.empty())
    y=trial.scale*s.sample_shape(s.models[s.nearest(trial.origin.data())],rng);
  else if(trial.family==ProposalFamily::Difference) {
    for(int j=0;j<d;++j) y[j]=trial.direction[j]+.01*normal(rng);
    const double length=y.norm();
    if(length>0) y*=trial.scale*std::sqrt(double(d))/length;
    else y.setZero();
  } else for(int j=0;j<d;++j) y[j]=trial.scale*normal(rng);
  for(int j=0;j<d;++j) delta[j]=float(y[j]);
}
}  // namespace fg::optimization
