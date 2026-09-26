#pragma once
#include "fractal/populations.hpp"
#include "fractal/wave.hpp"

namespace fg::fractal {
// The codec owns backend references as well as raw state (e.g. token records).
// It must reject a payload it cannot safely materialize in this environment.
template<class State,class Backend,class ActionPolicy>
class WavePopulationMember final : public PopulationMember {
 public:
  using Core=Wave<State,Backend,ActionPolicy>;
  using Save=std::function<std::vector<uint8_t>(const State&)>;
  using Load=std::function<void(State&,const std::vector<uint8_t>&)>;
  Core& core;
  std::string id,compatibility;
  std::function<void()> step;
  std::function<int()> elite_count;
  std::function<size_t()> exchange_count;
  std::function<RemovalPolicy()> removal;
  std::function<bool(int)> eligible;
  std::function<double(int)> score;
  bool imports_enabled=true;
  std::function<void()> after_commit;
  Save save_storage;
  Load load_storage;
  WavePopulationMember(Core& c,std::string member_id,std::string key,Save save,Load load)
      :core(c),id(std::move(member_id)),compatibility(std::move(key)),
       save_storage(std::move(save)),load_storage(std::move(load)) {}
  void advance() override {step();}
  MemberDescription describe() const override {
    MemberDescription d;d.id=id;d.compatibility=compatibility;d.exchange_count=exchange_count();d.imports_enabled=imports_enabled;
    for(int i=0;i<core.current.N;++i) {
      const auto& s=core.current;
      double rank=removal()==RemovalPolicy::VirtualReward && s.has_virtual_rewards?s.virtual_rewards[i]:s.rewards[i];
      d.rows.push_back({score?score(i):s.rewards[i],rank,s.alive(i) && (!eligible || eligible(i)),
                        core.has_elite && i<elite_count()});
    }
    return d;
  }
  WalkerPacket export_walker(int row) const override {
    if(row<0 || row>=core.current.N) throw std::invalid_argument("Invalid exported walker");
    State one;core.prepare(one,1,core.current.obs_dim,core.current.action_dim,core.current.has_infos);
    core.copy_row(core.current,row,one,0);one.has_virtual_rewards=core.current.has_virtual_rewards;
    CheckpointWriter out;out.scalar(uint32_t(1));
    out.scalar(one.obs_dim);out.scalar(one.action_dim);out.scalar(uint32_t(one.has_infos));
    auto storage=save_storage(one);out.vector(storage);
    auto write=[&](const auto& values){out.vector(values);};
    fields(one,write);out.scalar(uint32_t(one.has_virtual_rewards));
    auto branch=core.tree.export_branch(core.current.lineage[row]);branch.save_checkpoint(out);
    CheckpointWriter identity;identity.vector(storage);identity.scalar(one.rewards[0]);identity.scalar(one.step_rewards[0]);
    WalkerPacket p;p.source=id;p.compatibility=compatibility;p.row=row;
    p.score=score?score(row):one.rewards[0];
    p.identity=compatibility+":"+std::to_string(checkpoint_hash(identity.data.data(),identity.data.size()));
    p.bytes=std::move(out.data);return p;
  }
  void stage(const std::vector<WalkerImport>& imports) override {
    discard();if(imports.empty()) return;
    auto staged=std::make_unique<Core>(core.backend_,core.policy_);
    staged->reset(core.current.N,core.current.obs_dim,core.current.action_dim,core.current.has_infos);
    staged->current.has_virtual_rewards=core.current.has_virtual_rewards;
    for(int i=0;i<core.current.N;++i) staged->copy_row(core.current,i,staged->current,i);
    staged->tree=core.tree;staged->metrics=core.metrics;
    if(core.has_elite) {
      staged->prepare(staged->elite,core.elite.N,core.elite.obs_dim,core.elite.action_dim,core.elite.has_infos);
      for(int i=0;i<core.elite.N;++i) staged->copy_row(core.elite,i,staged->elite,i);
      staged->elite.has_virtual_rewards=core.elite.has_virtual_rewards;staged->has_elite=true;
    }
    std::set<int> seen;
    for(const auto& item:imports) {
      const auto& p=item.walker;int j=item.destination;
      if(p.compatibility!=compatibility || p.source==id || j<0 || j>=core.current.N ||
         (core.has_elite && j<elite_count()) || !seen.insert(j).second)
        throw std::invalid_argument("Invalid walker import");
      CheckpointReader in(p.bytes.data(),p.bytes.size());
      if(in.scalar<uint32_t>()!=1 || in.scalar<int32_t>()!=core.current.obs_dim ||
         in.scalar<int32_t>()!=core.current.action_dim || in.scalar<uint32_t>()!=uint32_t(core.current.has_infos))
        throw std::invalid_argument("Incompatible walker layout");
      State one;staged->prepare(one,1,core.current.obs_dim,core.current.action_dim,core.current.has_infos);
      load_storage(one,in.vector<uint8_t>());
      auto read=[&](auto& values) {
        using T=typename std::decay_t<decltype(values)>::value_type;
        auto v=in.template vector<T>();
        if(v.size()!=values.size()) throw std::invalid_argument("Invalid walker field shape");
        if constexpr(std::is_floating_point_v<T>)
          for(auto x:v) if(!std::isfinite(x)) throw std::invalid_argument("Nonfinite imported metadata");
        values=std::move(v);
      };
      fields(one,read);auto virtual_reward=in.scalar<uint32_t>();
      if(virtual_reward>1 || one.dones[0]>1 || one.truncated[0]>1 || one.recoverable[0]>1 ||
         one.dt[0]<1 || one.actual_dt[0]<0 || one.actual_dt[0]>one.dt[0] || !one.alive(0))
        throw std::invalid_argument("Invalid imported transition");
      one.has_virtual_rewards=virtual_reward;
      ExplorationTree branch;branch.load_checkpoint(in);in.finish();
      one.lineage[0]=staged->tree.import_branch(branch);
      staged->copy_row(one,0,staged->current,j);
    }
    staged->refresh_after_import(elite_count());pending_=std::move(staged);
  }
  void commit() noexcept override {
    if(!pending_) return;
    using std::swap;
    swap(core.current,pending_->current);swap(core.elite,pending_->elite);
    swap(core.tree,pending_->tree);core.has_elite=pending_->has_elite;core.metrics=pending_->metrics;
    std::iota(core.sources.begin(),core.sources.end(),0);
    std::iota(core.companions_.begin(),core.companions_.end(),0);
    std::iota(core.fitness_companions_.begin(),core.fitness_companions_.end(),0);
    std::fill(core.mask_.begin(),core.mask_.end(),0);
    discard();
    if(after_commit) after_commit();
  }
  void discard() noexcept override {pending_.reset();}
 private:
  std::unique_ptr<Core> pending_;
  template<class F> static void fields(State& s,F& f) {
    f(s.observations);f(s.rewards);f(s.step_rewards);f(s.virtual_rewards);
    f(s.dones);f(s.truncated);f(s.recoverable);f(s.actions);f(s.root_actions);
    f(s.dt);f(s.actual_dt);f(s.infos);
  }
};
} // namespace fg::fractal
