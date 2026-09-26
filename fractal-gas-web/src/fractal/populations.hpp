#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "thread_pool.hpp"
#include "fractal/population_control.hpp"

namespace fg::fractal {
// Owned, backend-specific transfer bytes. A compatibility key describes the task,
// state ABI and reward semantics, not the member's exploration parameters.
struct WalkerPacket {
  std::string source, identity, compatibility;
  int row = 0;
  double score = 0;
  std::vector<uint8_t> bytes;
};
struct ExchangeRow {
  double score = 0, removal_score = 0;
  bool eligible = false, protected_elite = false;
};
struct MemberDescription {
  std::string id, compatibility;
  size_t exchange_count = 0;
  bool imports_enabled = true;
  std::vector<ExchangeRow> rows;
};
struct MemberFrame {
  MemberDescription description;
  std::vector<WalkerPacket> exports;
  std::vector<int> destinations;
  std::vector<uint8_t> export_flags, import_flags;
};
struct WalkerImport { int destination; WalkerPacket walker; };
struct MemberExchange {
  std::string id;
  std::vector<WalkerImport> imports;
  size_t shortfall = 0;
};
class PopulationMember {
 public:
  virtual ~PopulationMember() = default;
  virtual void advance() = 0;
  virtual MemberDescription describe() const = 0;
  virtual WalkerPacket export_walker(int row) const = 0;
  // Stage owns all allocations and validation. Commit must not throw.
  virtual void stage(const std::vector<WalkerImport>&) = 0;
  virtual void commit() noexcept = 0;
  virtual void discard() noexcept = 0;
};
using ExportStrategy = std::function<std::vector<int>(const MemberDescription&)>;
using ImportStrategy = std::function<std::vector<int>(const MemberDescription&, const std::vector<int>&)>;
using DonorStrategy = std::function<std::vector<size_t>(const std::vector<WalkerPacket>&,
                                                      const std::string&, size_t, std::mt19937_64&)>;
inline std::vector<int> elite_exports(const MemberDescription& d) {
  std::vector<int> rows;
  for (size_t i=0;i<d.rows.size();++i)
    if(d.rows[i].eligible && std::isfinite(d.rows[i].score)) rows.push_back(int(i));
  std::stable_sort(rows.begin(),rows.end(),[&](int a,int b){
    if(d.rows[a].protected_elite!=d.rows[b].protected_elite)return d.rows[a].protected_elite;
    if(d.rows[a].protected_elite)return a<b;
    return d.rows[a].score>d.rows[b].score;
  });
  if(rows.size()>d.exchange_count) rows.resize(d.exchange_count);
  return rows;
}
inline std::vector<int> worst_imports(const MemberDescription& d,const std::vector<int>& exports) {
  std::vector<int> rows;
  for(size_t i=0;i<d.rows.size();++i)
    if(!d.rows[i].protected_elite && std::find(exports.begin(),exports.end(),int(i))==exports.end())
      rows.push_back(int(i));
  const auto retained=retention_order(int(d.rows.size()),[&](int i){return d.rows[i].removal_score;});
  std::vector<int> rank(d.rows.size());
  for(size_t i=0;i<retained.size();++i)rank[retained[i]]=int(i);
  std::sort(rows.begin(),rows.end(),[&](int a,int b){return rank[a]>rank[b];});
  if(rows.size()<d.exchange_count) throw std::invalid_argument("Insufficient unprotected import slots: "+d.id);
  rows.resize(d.exchange_count);
  return rows;
}
inline std::vector<size_t> random_foreign(const std::vector<WalkerPacket>& pool,
                                        const std::string& id,size_t count,std::mt19937_64& rng) {
  std::vector<size_t> rows;
  for(size_t i=0;i<pool.size();++i) if(pool[i].source!=id) rows.push_back(i);
  if(rows.size()<count) return {};
  // Rejection sampling avoids modulo bias and standard-library shuffle differences.
  for(size_t i=0;i<count;++i) {
    uint64_t n=rows.size()-i,limit=uint64_t(-n)%n,x;
    do{x=rng();}while(x<limit);
    std::swap(rows[i],rows[i+size_t(x%n)]);
  }
  rows.resize(count);return rows;
}
struct ExchangeStrategies {
  ExportStrategy exports=elite_exports;
  ImportStrategy imports=worst_imports;
  DonorStrategy donors=random_foreign;
};
class PopulationController {
 public:
  uint64_t rounds=0, exchanges=0;
  bool failed=false;
  size_t exchange_every=1, elite_capacity=20;
  ExchangeStrategies strategies;
  std::vector<WalkerPacket> pool, elites;
  std::vector<MemberFrame> frames;
  std::vector<MemberExchange> last_exchange;
  explicit PopulationController(uint64_t seed=7,size_t interval=1,size_t capacity=20)
      :exchange_every(interval),elite_capacity(capacity),random_(seed) {
    if(!interval) throw std::invalid_argument("Exchange interval must be positive");
  }
  bool due() const {return (rounds+1)%exchange_every==0;}
  MemberFrame capture(const PopulationMember& member,bool imports_enabled=true) const {
    MemberFrame f;f.description=member.describe();const auto& d=f.description;
    f.description.imports_enabled &= imports_enabled;
    auto selected=strategies.exports(d);
    if(d.imports_enabled) f.destinations=strategies.imports(d,selected);
    f.export_flags.assign(d.rows.size(),0);f.import_flags.assign(d.rows.size(),0);
    for(int i:selected) {
      if(i<0 || size_t(i)>=d.rows.size() || f.export_flags[i] || !d.rows[i].eligible)
        throw std::invalid_argument("Invalid export selection");
      f.export_flags[i]=1;auto p=member.export_walker(i);
      p.source=d.id;p.row=i;p.compatibility=d.compatibility;p.score=d.rows[i].score;
      f.exports.push_back(std::move(p));
    }
    for(int i:f.destinations) {
      if(i<0 || size_t(i)>=d.rows.size() || f.export_flags[i] || f.import_flags[i] || d.rows[i].protected_elite)
        throw std::invalid_argument("Invalid import selection");
      f.import_flags[i]=1;
    }
    return f;
  }
  // Also used by a WASM coordinator receiving frames from independent workers.
  void plan(std::vector<MemberFrame> incoming) {
    if(incoming.empty()) throw std::invalid_argument("Population requires at least one member");
    std::set<std::string> ids;
    const auto key=incoming.front().description.compatibility;
    for(const auto& f:incoming) {
      if(f.description.id.empty() || !ids.insert(f.description.id).second || key.empty() || f.description.compatibility!=key)
        throw std::invalid_argument("Incompatible or duplicate population members");
      for(const auto& p:f.exports)
        if(p.source!=f.description.id || p.compatibility!=key || !std::isfinite(p.score))
          throw std::invalid_argument("Invalid export packet");
    }
    frames=std::move(incoming);pool.clear();last_exchange.clear();
    for(const auto& f:frames) pool.insert(pool.end(),f.exports.begin(),f.exports.end());
    for(const auto& p:pool) {
      if(std::none_of(elites.begin(),elites.end(),[&](const auto& e){return e.identity==p.identity;})) elites.push_back(p);
    }
    std::stable_sort(elites.begin(),elites.end(),[](const auto& a,const auto& b){return a.score>b.score;});
    if(elites.size()>elite_capacity) elites.resize(elite_capacity);
    for(const auto& f:frames) {
      MemberExchange e;e.id=f.description.id;
      if(frames.size()>1) {
        auto donors=strategies.donors(pool,e.id,f.destinations.size(),random_);
        if(donors.size()!=f.destinations.size()) e.shortfall=f.destinations.size();
        else {
          std::set<size_t> seen;
          for(size_t i=0;i<donors.size();++i) {
            auto j=donors[i];
            if(j>=pool.size() || pool[j].source==e.id || !seen.insert(j).second)
              throw std::invalid_argument("Invalid foreign donor selection");
            e.imports.push_back({f.destinations[i],pool[j]});
          }
        }
      }
      last_exchange.push_back(std::move(e));
    }
    ++exchanges;
  }
  void advance(const std::vector<PopulationMember*>& members,ThreadPool* executor=nullptr) {
    if(failed) throw std::runtime_error("Population failed; reset before continuing");
    if(members.empty()) throw std::invalid_argument("Population requires members");
    // Validate configuration before advancing any member.
    std::set<std::string> ids;std::string key;
    for(auto* m:members) {
      auto d=m->describe();if(key.empty()) key=d.compatibility;
      if(!ids.insert(d.id).second || d.compatibility!=key) throw std::invalid_argument("Incompatible population");
      if(members.size()>1 && d.imports_enabled) strategies.imports(d,strategies.exports(d));
    }
    try {
    if(executor) executor->parallel_for(int(members.size()),[&](int i,int){members[i]->advance();});
    else for(auto* m:members) m->advance();
    if(due()) {
      auto prepared=*this;
      std::vector<MemberFrame> next;
      for(auto* m:members) next.push_back(prepared.capture(*m,members.size()>1));
      prepared.plan(std::move(next));
      try {for(size_t i=0;i<members.size();++i) members[i]->stage(prepared.last_exchange[i].imports);}
      catch(...) {for(auto* m:members)m->discard();throw;}
      for(auto* m:members)m->commit();
      *this=std::move(prepared);
    }
    ++rounds;
    } catch(...) { failed=true; throw; }
  }
 private:
  std::mt19937_64 random_;
};
} // namespace fg::fractal
