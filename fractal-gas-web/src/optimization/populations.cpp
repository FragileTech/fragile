#include "optimization/populations.hpp"
#include <set>
#include "fractal/euclidean.hpp"
#include "fractal/checkpoint.hpp"

namespace fg::optimization {
namespace {
Json object(){Json j;j.kind=Json::Object;return j;}
Json list(){Json j;j.kind=Json::Array;return j;}
Json str(const std::string& s){Json j;j.kind=Json::String;j.string=s;return j;}
Json flag(bool b){Json j;j.kind=Json::Boolean;j.number=b;return j;}
std::string hex(const std::vector<uint8_t>& data){
  static const char* chars="0123456789abcdef";std::string s;s.reserve(data.size()*2);
  for(auto b:data){s.push_back(chars[b>>4]);s.push_back(chars[b&15]);}return s;
}
std::vector<uint8_t> unhex(const std::string& s){
  if(s.size()%2 || s.size()>256*1024*1024) throw std::invalid_argument("Invalid transfer size");
  auto digit=[](char c){if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::invalid_argument("Invalid transfer encoding");};
  std::vector<uint8_t> bytes;bytes.reserve(s.size()/2);
  for(size_t i=0;i<s.size();i+=2) bytes.push_back(uint8_t(digit(s[i])*16+digit(s[i+1])));
  return bytes;
}
fractal::WalkerPacket packet_read(const Json& j){
  fractal::WalkerPacket p;p.source=j["source"].str();p.identity=j["identity"].str();
  p.compatibility=j["compatibility"].str();p.row=integer(j["row"],0,0,100000,"export row");
  p.score=bounded(j["score"],0,-1e300,1e300,"export score");p.bytes=unhex(j["payload"].str());return p;
}
Json integers(const std::vector<int>& v){auto j=list();for(auto x:v)j.array.push_back(number(x));return j;}
}
Json packet_json(const fractal::WalkerPacket& p,bool payload){
  auto j=object();j.object["source"]=str(p.source);j.object["identity"]=str(p.identity);
  j.object["row"]=number(p.row);j.object["score"]=number(p.score);
  if(payload){j.object["compatibility"]=str(p.compatibility);j.object["payload"]=str(hex(p.bytes));}return j;
}
Json exchange_frame_json(const fractal::MemberFrame& f){
  auto j=object();j.object["id"]=str(f.description.id);j.object["compatibility"]=str(f.description.compatibility);
  j.object["exchange_count"]=number(f.description.exchange_count);j.object["imports_enabled"]=flag(f.description.imports_enabled);j.object["rows"]=list();
  for(const auto& r:f.description.rows){auto v=object();v.object["score"]=number(r.score);v.object["removal_score"]=number(r.removal_score);v.object["eligible"]=flag(r.eligible);v.object["protected"]=flag(r.protected_elite);j.object["rows"].array.push_back(v);}
  j.object["exports"]=list();for(const auto& p:f.exports)j.object["exports"].array.push_back(packet_json(p));
  j.object["destinations"]=integers(f.destinations);return j;
}
fractal::MemberFrame exchange_frame_read(const Json& j){
  fractal::MemberFrame f;auto& d=f.description;d.id=j["id"].str();d.compatibility=j["compatibility"].str();
  d.exchange_count=integer(j["exchange_count"],0,0,100000,"exchange count");d.imports_enabled=j["imports_enabled"].flag(true);
  for(const auto& r:j["rows"].items())d.rows.push_back({r["score"].num(-INFINITY),r["removal_score"].num(-INFINITY),r["eligible"].flag(),r["protected"].flag()});
  f.export_flags.assign(d.rows.size(),0);f.import_flags.assign(d.rows.size(),0);
  for(const auto& p:j["exports"].items()){
    auto packet=packet_read(p);int row=packet.row;
    if(size_t(row)>=d.rows.size() || f.export_flags[row] || !d.rows[row].eligible)throw std::invalid_argument("Invalid exported row");
    f.export_flags[row]=1;f.exports.push_back(std::move(packet));
  }
  for(const auto& v:j["destinations"].items()){
    int row=integer(v,-1,0,int(d.rows.size())-1,"import destination");
    if(f.export_flags[row] || f.import_flags[row] || d.rows[row].protected_elite)throw std::invalid_argument("Invalid import destination");
    f.import_flags[row]=1;f.destinations.push_back(row);
  }
  if(f.destinations.size()!=(d.imports_enabled?d.exchange_count:0) || f.exports.size()>d.exchange_count)throw std::invalid_argument("Invalid exchange count");
  return f;
}
Json exchange_imports_json(const std::vector<fractal::WalkerImport>& imports){
  auto j=list();for(const auto& i:imports){auto v=object();v.object["destination"]=number(i.destination);v.object["walker"]=packet_json(i.walker);j.array.push_back(v);}return j;
}
std::vector<fractal::WalkerImport> exchange_imports_read(const Json& j){
  std::vector<fractal::WalkerImport> result;for(const auto& v:j.items())result.push_back({integer(v["destination"],-1,0,100000,"destination"),packet_read(v["walker"])});return result;
}
PopulationExperiment::PopulationExperiment(const Json& input,bool remote)
 :controller(uint64_t(integer(input["seed"],7,0,2147483647,"population seed")),
             size_t(integer(input["exchange_every"],1,1,1000000,"exchange interval")),
             size_t(integer(input["global_elites"],20,0,100000,"global elites"))),remote_(remote) {
  budget_=uint64_t(bounded(input["max_evaluations"],0,0,1e12,"population budget"));
  if(double(budget_)!=input["max_evaluations"].num(0))throw std::invalid_argument("Budget must be an integer");
  Json defaults=input["defaults"];if(defaults.kind==Json::Null)defaults=object();
  if(defaults.kind!=Json::Object)throw std::invalid_argument("Shared defaults must be an object");
  if(defaults["algorithm"].kind==Json::Null)defaults.object["algorithm"]=str("wave");
  if(defaults["walkers"].kind==Json::Null)defaults.object["walkers"]=number(256);
  if(defaults["elites"].kind==Json::Null)defaults.object["elites"]=number(5);
  auto members=input["members"];
  if(members.kind==Json::Null){members=list();for(int i=0;i<4;++i){auto m=object();m.object["id"]=str("swarm-"+std::to_string(i+1));members.array.push_back(m);}}
  if(members.kind!=Json::Array || members.array.empty())throw std::invalid_argument("Population requires a nonempty members list");
  concurrency_=size_t(integer(input["concurrency"],std::min(4,int(members.array.size())),1,int(members.array.size()),"execution slots"));
  config=object();config.object["version"]=number(1);config.object["seed"]=number(input["seed"].num(7));
  config.object["exchange_every"]=number(controller.exchange_every);config.object["global_elites"]=number(controller.elite_capacity);
  config.object["max_evaluations"]=number(budget_);config.object["concurrency"]=number(concurrency_);config.object["members"]=list();
  std::set<std::string> ids;std::string task_key;uint64_t initial=0,memory=32*1024*1024;
  for(const auto& member:members.array){
    if(member.kind!=Json::Object || (member["settings"].kind!=Json::Null && member["settings"].kind!=Json::Object))
      throw std::invalid_argument("Each member must contain an object of settings");
    const auto id=member["id"].str();if(id.empty() || !ids.insert(id).second)throw std::invalid_argument("Swarm IDs must be nonempty and unique");
    Json resolved=defaults;
    for(const auto& field:member["settings"].object)resolved.object[field.first]=field.second;
    if(member["settings"]["seed"].kind==Json::Null){
      auto h=fractal::checkpoint_hash(reinterpret_cast<const uint8_t*>(id.data()),id.size());
      resolved.object["seed"]=number(fractal::descendant_lineage(uint64_t(config["seed"].num()),h)&0x7fffffffULL);
    }
    resolved.object["max_evaluations"]=number(budget_);
    Benchmark b(resolved);Settings s(b.config);
    if(s.algorithm!="wave")throw std::invalid_argument("Fractal Populations currently supports Wave members");
    int count=integer(member["exchange_count"],s.elites,0,s.walkers,"exchange count");
    if(members.array.size()>1 && s.walkers<std::max(s.elites,count)+count)throw std::invalid_argument("Not enough walkers for protected elites and imports");
    BasinArchive archive(b,s.periodic);
    auto key=stringify(JsonReader(archive.export_json()).read()["compatibility"]);
    if(task_key.empty())task_key=key;else if(task_key!=key)throw std::invalid_argument("Members must share the same task and scoring semantics");
    initial+=Session::initial_evaluation_bound(s.json);
    memory+=32*1024*1024ULL+uint64_t(s.max_walkers)*(uint64_t(b.d)*32+1024);
    if(memory>1024*1024*1024ULL)throw std::invalid_argument("Population exceeds 1 GiB resource budget; reduce swarm count or sizes");
    s.json.object["population_imports"]=number(members.array.size()>1?count:0);
    auto entry=object();entry.object["id"]=str(id);entry.object["settings"]=s.json;entry.object["exchange_count"]=number(count);
    config.object["members"].array.push_back(entry);
  }
  if(budget_ && initial>budget_)throw std::invalid_argument("Shared budget cannot initialize every swarm");
  task_=std::make_unique<Benchmark>(config["members"].array[0]["settings"]);
  archive_=std::make_unique<BasinArchive>(*task_,Settings(task_->config).periodic);
  config.object["initial_evaluations_upper_bound"]=number(initial);
  if(!remote_){
    for(const auto& m:config["members"].array){
      sessions.push_back(std::make_unique<Session>(m["settings"]));
      sessions.back()->exchange_member(m["id"].str(),int(m["exchange_count"].num()));
    }
    executor_=std::make_unique<ThreadPool>(int(concurrency_));
    auto reports=list();for(size_t i=0;i<sessions.size();++i)reports.array.push_back(report(i));
    accept_reports(reports,true);
  }
}
Json PopulationExperiment::report(size_t i){
  auto& session=*sessions[i];const auto& m=config["members"].array[i];
  auto member=session.exchange_member(m["id"].str(),int(m["exchange_count"].num()));
  auto r=object();r.object["id"]=m["id"];r.object["evaluations"]=number(session.benchmark.evaluations);
  r.object["next_evaluations"]=number(session.next_evaluations());r.object["settings"]=session.settings.json;
  r.object["events"]=session.take_basin_events();
  r.object["frame"]=exchange_frame_json(controller.capture(*member));return r;
}
uint64_t PopulationExperiment::evaluations()const{uint64_t n=0;for(const auto& r:reports_)n+=uint64_t(r["evaluations"].num());return n;}
uint64_t PopulationExperiment::next_cost()const{uint64_t n=0;for(const auto& r:reports_)n+=uint64_t(r["next_evaluations"].num());return n;}
void PopulationExperiment::admit()const{
  if(failed_)throw std::runtime_error("Population failed; reset before continuing");
  if(!initialized_)throw std::logic_error("Population is not initialized");
  if(round_pending_)throw std::logic_error("An exchange is awaiting commit");
  if(budget_ && (evaluations()>budget_ || next_cost()>budget_-evaluations()))throw std::runtime_error("Shared budget cannot admit the next complete round");
}
void PopulationExperiment::accept_reports(const Json& reports,bool initial){
  if(reports.array.size()!=config["members"].array.size())throw std::invalid_argument("Incomplete population round");
  std::vector<Json> ordered;
  for(size_t i=0;i<config["members"].array.size();++i){
    const auto id=config["members"].array[i]["id"].str();
    auto found=std::find_if(reports.array.begin(),reports.array.end(),[&](const auto& r){return r["id"].str()==id;});
    if(found==reports.array.end())throw std::invalid_argument("Missing swarm report: "+id);
    auto frame=exchange_frame_read((*found)["frame"]);
    if(frame.description.id!=id)throw std::invalid_argument("Mismatched swarm frame");
    auto expected="optimization-wave-v1:"+stringify(JsonReader(archive_->export_json()).read()["compatibility"]);
    if(frame.description.compatibility!=expected)throw std::invalid_argument("Incompatible population task");
    const auto eval=bounded((*found)["evaluations"],0,0,1e12,"member evaluations");
    const auto next=bounded((*found)["next_evaluations"],0,0,1e12,"next evaluations");
    if(std::floor(eval)!=eval || std::floor(next)!=next || (!initial && i<reports_.size() && eval<reports_[i]["evaluations"].num()))
      throw std::invalid_argument("Invalid member evaluation accounting");
    config.object["members"].array[i].object["settings"]=(*found)["settings"];
    for(const auto& event:(*found)["events"].array) {
      auto event_key=id+":"+std::to_string(integer(event["member_round"],0,0,1000000000,"member round"));
      if(seen_events_.insert(event_key).second) {
        auto evidence=event;
        for(auto& entry:evidence.object["entries"].array) {
          entry.object["settings"].object["population_member"]=str(id);
          entry.object["settings"].object["population_member_round"]=event["member_round"];
        }
        archive_->merge_event(evidence,++basin_round_);
      }
    }
    ordered.push_back(*found);
  }
  reports_=std::move(ordered);initialized_=true;
  if(budget_ && evaluations()>budget_)throw std::runtime_error("Population exceeded its admitted evaluation budget");
  if(initial){
    std::vector<fractal::MemberFrame> frames;for(const auto& r:reports_)frames.push_back(exchange_frame_read(r["frame"]));
    // Validate task compatibility without counting initialization as an exchange.
    const auto key=frames.front().description.compatibility;
    for(const auto& f:frames)if(f.description.compatibility!=key)throw std::invalid_argument("Incompatible members");
  }
}
Json PopulationExperiment::finish_round(const Json& reports){
  accept_reports(reports,false);
  pending_controller_=std::make_unique<fractal::PopulationController>(controller);
  auto plans=list();
  if(controller.due()){
    std::vector<fractal::MemberFrame> frames;for(const auto& r:reports_)frames.push_back(exchange_frame_read(r["frame"]));
    pending_controller_->plan(std::move(frames));
    for(const auto& e:pending_controller_->last_exchange){auto j=object();j.object["id"]=str(e.id);j.object["imports"]=exchange_imports_json(e.imports);plans.array.push_back(j);}
  }
  ++pending_controller_->rounds;round_pending_=true;
  auto result=object();result.object["plans"]=plans;return result;
}
void PopulationExperiment::step(){
  if(remote_)throw std::logic_error("Remote populations advance through worker reports");
  admit();
  bool dispatched=false;
  try {
    const auto archive=archive_->export_json();for(auto& s:sessions)s->synchronize_basins(archive);
    // Synchronized archive evidence can change the admission bound of a restart.
    uint64_t required=0;for(auto& s:sessions)required+=s->next_evaluations();
    if(budget_ && required>budget_-evaluations())throw std::runtime_error("Shared budget cannot admit synchronized restarts");
    dispatched=true;
    executor_->parallel_for(int(sessions.size()),[&](int i,int){
      try { sessions[i]->step(); }
      catch(const std::exception& e) {
        throw std::runtime_error("Swarm "+config["members"].array[i]["id"].str()+": "+e.what());
      }
    });
    auto reports=list();for(size_t i=0;i<sessions.size();++i)reports.array.push_back(report(i));
    finish_round(reports);
    std::vector<std::unique_ptr<fractal::PopulationMember>> members;
    try {
      if(controller.due())for(size_t i=0;i<sessions.size();++i){
        const auto& m=config["members"].array[i];members.push_back(sessions[i]->exchange_member(m["id"].str(),int(m["exchange_count"].num())));
        members.back()->stage(pending_controller_->last_exchange[i].imports);
      }
    }catch(...){for(auto& m:members)m->discard();throw;}
    for(auto& m:members)m->commit();
    controller=std::move(*pending_controller_);pending_controller_.reset();round_pending_=false;
    for(auto& s:sessions)s->refresh_exchange();
  }catch(...){
    if(dispatched) {
      failed_=true;
      for(size_t i=0;i<sessions.size() && i<reports_.size();++i)
        reports_[i].object["evaluations"]=number(sessions[i]->benchmark.evaluations);
    }
    throw;
  }
}
Json PopulationExperiment::status()const{
  auto j=object();j.object["round"]=number(controller.rounds);j.object["exchanges"]=number(controller.exchanges);
  j.object["evaluations"]=number(evaluations());j.object["next_evaluations"]=number(next_cost());
  j.object["budget_exhausted"]=flag(budget_ && (evaluations()>budget_ || next_cost()>budget_-evaluations()));
  j.object["evaluations_is_lower_bound"]=flag(remote_ && failed_);
  j.object["failed"]=flag(failed_);j.object["global_elites"]=list();
  for(const auto& e:controller.elites)j.object["global_elites"].array.push_back(packet_json(e,false));
  j.object["basins"]=archive_->summary();j.object["members"]=list();
  for(size_t i=0;i<reports_.size();++i){
    auto m=object();m.object["id"]=reports_[i]["id"];m.object["settings"]=reports_[i]["settings"];
    m.object["evaluations"]=reports_[i]["evaluations"];m.object["imports"]=list();m.object["exports"]=list();
    if(i<controller.frames.size())for(const auto& p:controller.frames[i].exports)m.object["exports"].array.push_back(number(p.row));
    if(i<controller.last_exchange.size()){
      const auto& e=controller.last_exchange[i];m.object["shortfall"]=number(e.shortfall);
      for(const auto& imp:e.imports){auto v=packet_json(imp.walker,false);v.object["destination"]=number(imp.destination);m.object["imports"].array.push_back(v);}
    }
    j.object["members"].array.push_back(m);
  }
  return j;
}
Json PopulationExperiment::request(const Json& req){
  auto op=req["op"].str();
  if(op=="config")return config;
  if(op=="status")return status();
  if(op=="initialize"){if(initialized_)throw std::logic_error("Population already initialized");accept_reports(req["reports"],true);return status();}
  if(op=="prepare"){admit();auto j=object();j.object["archive"]=str(archive_->export_json());return j;}
  if(op=="refresh"){accept_reports(req["reports"],false);return status();}
  if(op=="admit"){accept_reports(req["reports"],false);admit();authorized_=true;return status();}
  if(op=="finish"){if(!authorized_ || round_pending_)throw std::logic_error("Admit a round before finishing it");authorized_=false;return finish_round(req["reports"]);}
  if(op=="commit"){
    if(!round_pending_ || !pending_controller_)throw std::logic_error("No pending exchange");
    controller=std::move(*pending_controller_);pending_controller_.reset();round_pending_=false;return status();
  }
  if(op=="fail"){
    failed_=true;
    for(const auto& report:req["reports"].array)
      for(auto& prior:reports_)if(prior["id"].str()==report["id"].str())
        prior.object["evaluations"]=number(std::max(prior["evaluations"].num(),report["evaluations"].num()));
    return status();
  }
  if(op=="snapshot") {
    if(remote_)throw std::logic_error("Snapshots belong to member workers");
    int i=integer(req["member"],0,0,int(sessions.size())-1,"member index");return array(sessions[i]->snapshot);
  }
  if(op=="settings") {
    if(remote_)throw std::logic_error("Member settings belong to member workers");
    if(round_pending_ || failed_)throw std::logic_error("Population is not at a settings boundary");
    int i=integer(req["member"],0,0,int(sessions.size())-1,"member index");
    const auto& patch=req["patch"];
    for(const auto& field:patch.object)
      if(field.first=="boundary" || field.first=="periodic" || field.first=="max_evaluations")
        throw std::invalid_argument("Task geometry and shared budget changes require population reset");
    auto next=sessions[i]->preview_settings(patch);int imports=int(config["members"].array[i]["exchange_count"].num());
    if(sessions.size()>1 && next["walkers"].num()<std::max(next["elites"].num(),double(imports))+imports)
      throw std::invalid_argument("Insufficient import slots after settings update");
    sessions[i]->update_settings(patch);reports_[i]=report(i);
    config.object["members"].array[i].object["settings"]=sessions[i]->settings.json;return status();
  }
  if(op=="step"){step();return status();}
  throw std::invalid_argument("Unknown population operation");
}
} // namespace fg::optimization
