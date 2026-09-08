#include "optimization/benchmark.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>

namespace fg::optimization {
Json number(double v) {
  Json j;
  j.kind = Json::Number;
  j.number = v;
  return j;
}
Json array(const std::vector<double>& v) {
  Json j;
  j.kind = Json::Array;
  for (auto x : v) j.array.push_back(number(x));
  return j;
}
std::string stringify(const Json& j) {
  std::ostringstream o;
  o.imbue(std::locale::classic());
  o << std::setprecision(17);
  switch (j.kind) {
    case Json::Null:
      return "null";
    case Json::Boolean:
      return j.number ? "true" : "false";
    case Json::Number:
      if (!std::isfinite(j.number)) return "null";
      o << j.number;
      break;
    case Json::String:
      o << '"';
      for (unsigned char c : j.string) {
        if (c == '"' || c == '\\')
          o << '\\' << c;
        else if (c < 32)
          o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c)
            << std::dec;
        else
          o << c;
      }
      o << '"';
      break;
    case Json::Array:
      o << '[';
      for (size_t i = 0; i < j.array.size(); ++i) {
        if (i) o << ',';
        o << stringify(j.array[i]);
      }
      o << ']';
      break;
    case Json::Object:
      o << '{';
      {
        bool first = true;
        for (auto& kv : j.object) {
          if (!first) o << ',';
          first = false;
          Json key;
          key.kind = Json::String;
          key.string = kv.first;
          o << stringify(key) << ':' << stringify(kv.second);
        }
      }
      o << '}';
      break;
  }
  return o.str();
}
int integer(const Json& j, int fallback, int lo, int hi, const char* name) {
  double v = j.num(fallback);
  if (!std::isfinite(v) || v < lo || v > hi || std::floor(v) != v)
    throw std::invalid_argument(std::string("Invalid ") + name);
  return int(v);
}
double bounded(const Json& j, double fallback, double lo, double hi,
               const char* name) {
  double v = j.num(fallback);
  if (!std::isfinite(v) || v < lo || v > hi)
    throw std::invalid_argument(std::string("Invalid ") + name);
  return v;
}
double normal(Rng& rng) {
  const double radial = std::max(1e-12f, rng.uniform01());
  const double angular = rng.uniform01();
  return std::sqrt(-2 * std::log(radial)) * std::cos(2 * pi * angular);
}
const std::string& catalog_json() {
  static const std::string s =
      R"json({"version":"fgopt-2","algorithms":[{"id":"fmc","name":"FMC","velocity":false},{"id":"wave_jump","name":"Wave Jump","velocity":false},{"id":"euclidean","name":"Euclidean Gas","velocity":true},{"id":"wave","name":"Wave","velocity":false},{"id":"graph","name":"Graph","velocity":false}],"benchmarks":[
{"id":"sphere","name":"Sphere","bounds":[-1000,1000],"minDimension":1,"minimum":0,"gradient":"analytic"},
{"id":"quadratic","name":"Quadratic Well","bounds":[-10,10],"minDimension":1,"minimum":0,"parameters":{"alpha":0.1},"gradient":"analytic"},
{"id":"mexican_hat","name":"Mexican Hat","bounds":[-10,10],"minDimension":1,"parameters":{"lambda_h":0.13,"vev":246,"field_scale":246,"tilt":0},"reference":"Ring minima only when tilt is zero","gradient":"analytic"},
{"id":"rastrigin","name":"Rastrigin","bounds":[-5.12,5.12],"minDimension":1,"minimum":0,"gradient":"analytic"},
{"id":"eggholder","name":"EggHolder","bounds":[-512,512],"dimension":2,"minimum":-959.64066271,"gradient":"analytic; central difference at cusps"},
{"id":"styblinski_tang","name":"Styblinski–Tang","bounds":[-5,5],"minDimension":1,"reference":"Minimum approximately -39.16617 × dimension","gradient":"analytic"},
{"id":"rosenbrock","name":"Rosenbrock","bounds":[-10,10],"minDimension":2,"minimum":0,"gradient":"analytic"},
{"id":"easom","name":"Easom","bounds":[-100,100],"dimension":2,"minimum":-1,"gradient":"analytic"},
{"id":"holder_table","name":"Holder Table","bounds":[-10,10],"dimension":2,"minimum":-19.2085,"gradient":"analytic; central difference at cusps"},
{"id":"lennard_jones","name":"Lennard–Jones","bounds":[-15,15],"parameters":{"n_atoms":10},"dimensionRule":"3 × n_atoms","reference":"Known energies: 2 atoms -1; 3 atoms -3; 4 atoms -6; 10 atoms -28.422532","gradient":"analytic"},
{"id":"constant","name":"Constant","bounds":[-10,10],"minDimension":1,"minimum":0,"gradient":"zero"},
{"id":"stochastic_gaussian","name":"Stochastic Gaussian","bounds":[-10,10],"minDimension":1,"stochastic":true,"parameters":{"std":1},"reference":"Expected value 0; no spatial optimum","gradient":"disabled"},
{"id":"gaussian_mixture","name":"Mixture of Gaussians","bounds":[-10,10],"minDimension":1,"parameters":{"n_gaussians":3,"benchmark_seed":42},"reference":"Component centers are reference points, not guaranteed minima","gradient":"analytic"}
]})json";
  return s;
}
Benchmark::Benchmark(const Json& input) : config(input) {
  if (config.kind != Json::Object)
    throw std::invalid_argument("Configuration must be an object");
  id = config["benchmark"].str("rastrigin");
  Json name;
  name.kind = Json::String;
  name.string = id;
  config.object["benchmark"] = name;
  auto cat = JsonReader(catalog_json()).read();
  const Json* entry = nullptr;
  for (auto& b : cat["benchmarks"].array)
    if (b["id"].str() == id) entry = &b;
  if (!entry) throw std::invalid_argument("Unknown benchmark");
  d = integer(config["dimensions"], 3, 1, 4096, "dimensions");
  if ((*entry)["dimension"].kind != Json::Null &&
      d != (*entry)["dimension"].num())
    throw std::invalid_argument("This benchmark requires two dimensions");
  if (d < (*entry)["minDimension"].num(1))
    throw std::invalid_argument("Benchmark dimension is too small");
  if (id == "lennard_jones") {
    int atoms = integer(config["n_atoms"], 10, 2, 1365, "atom count");
    d = 3 * atoms;
    config.object["n_atoms"] = number(atoms);
  }
  config.object["dimensions"] = number(d);
  low = bounded(config["low"], (*entry)["bounds"].array[0].num(), -1e6, 1e6,
                "lower bound");
  high = bounded(config["high"], (*entry)["bounds"].array[1].num(), -1e6, 1e6,
                 "upper bound");
  if (high <= low)
    throw std::invalid_argument("Upper bound must exceed lower bound");
  config.object["low"] = number(low);
  config.object["high"] = number(high);
  alpha = bounded(config["alpha"], .1, 0, 1e6, "quadratic curvature");
  lambda = bounded(config["lambda_h"], .13, 0, 1e6, "quartic coupling");
  radius = bounded(config["vev"], 246, 0, 1e6, "vev") /
           bounded(config["field_scale"], 246, 1e-12, 1e12, "field scale");
  tilt = bounded(config["tilt"], 0, -1e6, 1e6, "tilt");
  stochastic = id == "stochastic_gaussian";
  stddev = bounded(config["std"], 1, 0, 1e6, "noise deviation");
  if (id == "gaussian_mixture") {
    components =
        integer(config["n_gaussians"], 3, 1, 256, "mixture component count");
    OptimizationRng rng(
        integer(config["benchmark_seed"], 42, 0, 2147483647, "benchmark seed"));
    auto read = [&](const char* key, std::vector<double>& dest, size_t size) {
      if (config[key].kind == Json::Null) return false;
      for (auto& v : config[key].items()) {
        if (v.kind == Json::Array)
          for (auto& w : v.items()) dest.push_back(w.num());
        else
          dest.push_back(v.num());
      }
      if (dest.size() != size)
        throw std::invalid_argument(std::string("Wrong shape for ") + key);
      return true;
    };
    if (!read("centers", centers, size_t(components) * d))
      for (int i = 0; i < components * d; ++i)
        centers.push_back(low + (high - low) * rng.uniform01());
    if (!read("stds", stds, centers.size()))
      for (size_t i = 0; i < centers.size(); ++i)
        stds.push_back(.1 + 1.9 * rng.uniform01());
    if (!read("weights", weights, components))
      weights.assign(components, 1.0 / components);
    double sum = 0;
    for (auto w : weights) {
      if (w < 0) throw std::invalid_argument("Negative mixture weight");
      sum += w;
    }
    if (sum <= 0)
      throw std::invalid_argument("Mixture weights must have positive sum");
    for (auto s : stds)
      if (s <= 0)
        throw std::invalid_argument("Mixture deviations must be positive");
    for (auto& w : weights) w /= sum;
    config.object["centers"] = array(centers);
    config.object["stds"] = array(stds);
    config.object["weights"] = array(weights);
    config.object["n_gaussians"] = number(components);
  }
}
double Benchmark::value(const std::vector<double>& x) const {
  double s = 0;
  if (id == "constant" || stochastic) return 0;
  if (id == "sphere" || id == "quadratic" || id == "mexican_hat") {
    for (auto v : x) s += v * v;
    return id == "sphere" ? s
           : id == "quadratic"
               ? .5 * alpha * s
               : .25 * lambda * std::pow(s - radius * radius, 2) - tilt * x[0];
  }
  if (id == "rastrigin") {
    for (auto v : x) s += v * v - 10 * std::cos(2 * pi * v) + 10;
    return s;
  }
  if (id == "styblinski_tang") {
    for (auto v : x) s += (v * v * v * v - 16 * v * v + 5 * v) / 2;
    return s;
  }
  if (id == "rosenbrock") {
    for (int k = 0; k < d - 1; ++k)
      s += 100 * std::pow(x[k] * x[k] - x[k + 1], 2) + std::pow(x[k] - 1, 2);
    return s;
  }
  if (id == "eggholder")
    return -(x[1] + 47) * std::sin(std::sqrt(std::abs(x[0] / 2 + x[1] + 47))) -
           x[0] * std::sin(std::sqrt(std::abs(x[0] - x[1] - 47)));
  if (id == "easom")
    return -std::cos(x[0]) * std::cos(x[1]) *
           std::exp(-std::pow(x[0] - pi, 2) - std::pow(x[1] - pi, 2));
  if (id == "holder_table")
    return -std::abs(std::sin(x[0]) * std::cos(x[1]) *
                     std::exp(std::abs(1 - std::hypot(x[0], x[1]) / pi)));
  if (id == "lennard_jones") {
    for (int a = 0; a < d; a += 3)
      for (int b = 0; b < a; b += 3) {
        double r2 = 0;
        for (int k = 0; k < 3; ++k) r2 += std::pow(x[a + k] - x[b + k], 2);
        if (r2 == 0) return INFINITY;
        double r6 = 1 / (r2 * r2 * r2);
        s += 4 * r6 * (r6 - 1);
      }
    return s;
  }
  std::vector<double> logs(components);
  double maximum = -INFINITY;
  for (int c = 0; c < components; ++c) {
    double logp = weights[c] > 0 ? std::log(weights[c]) : -INFINITY;
    for (int k = 0; k < d; ++k) {
      size_t j = size_t(c) * d + k;
      logp -= .5 * (std::log(2 * pi) + 2 * std::log(stds[j]) +
                    std::pow((x[k] - centers[j]) / stds[j], 2));
    }
    logs[c] = logp;
    maximum = std::max(maximum, logp);
  }
  for (auto l : logs) s += std::exp(l - maximum);
  return -(maximum + std::log(s));
}
double Benchmark::evaluate(const float* x, Rng* rng) const {
  for (int k = 0; k < d; ++k)
    if (!std::isfinite(x[k])) return INFINITY;
  if (stochastic && rng) return stddev * normal(*rng);
  return value(std::vector<double>(x, x + d));
}
void Benchmark::gradient(const float* p, float* out) const {
  std::vector<double> x(p, p + d), g(d, 0);
  double r2 = 0;
  for (auto v : x) r2 += v * v;
  if (id == "sphere" || id == "quadratic" || id == "mexican_hat" ||
      id == "rastrigin" || id == "styblinski_tang")
    for (int k = 0; k < d; ++k) {
      double v = x[k];
      g[k] = id == "sphere"      ? 2 * v
             : id == "quadratic" ? alpha * v
             : id == "mexican_hat"
                 ? lambda * (r2 - radius * radius) * v - (k == 0 ? tilt : 0)
             : id == "rastrigin" ? 2 * v + 20 * pi * std::sin(2 * pi * v)
                                 : 2 * v * v * v - 16 * v + 2.5;
    }
  else if (id == "rosenbrock")
    for (int k = 0; k < d - 1; ++k) {
      double e = x[k] * x[k] - x[k + 1];
      g[k] += 400 * x[k] * e + 2 * (x[k] - 1);
      g[k + 1] -= 200 * e;
    }
  else if (id == "easom") {
    double e = std::exp(-std::pow(x[0] - pi, 2) - std::pow(x[1] - pi, 2));
    g[0] = e * std::cos(x[1]) *
           (std::sin(x[0]) + 2 * (x[0] - pi) * std::cos(x[0]));
    g[1] = e * std::cos(x[0]) *
           (std::sin(x[1]) + 2 * (x[1] - pi) * std::cos(x[1]));
  } else if (id == "lennard_jones")
    for (int a = 0; a < d; a += 3)
      for (int b = 0; b < a; b += 3) {
        double q = 0;
        for (int k = 0; k < 3; ++k) q += std::pow(x[a + k] - x[b + k], 2);
        double r6 = 1 / (q * q * q), f = 24 * r6 * (1 - 2 * r6) / q;
        for (int k = 0; k < 3; ++k) {
          double v = f * (x[a + k] - x[b + k]);
          g[a + k] += v;
          g[b + k] -= v;
        }
      }
  else if (id == "gaussian_mixture") {
    double nll = value(x);
    for (int c = 0; c < components; ++c) {
      double l = weights[c] > 0 ? std::log(weights[c]) : -INFINITY;
      for (int k = 0; k < d; ++k) {
        size_t j = size_t(c) * d + k;
        l -= .5 * (std::log(2 * pi) + 2 * std::log(stds[j]) +
                   std::pow((x[k] - centers[j]) / stds[j], 2));
      }
      double probability = std::exp(l + nll);
      for (int k = 0; k < d; ++k) {
        size_t j = size_t(c) * d + k;
        g[k] += probability * (x[k] - centers[j]) / (stds[j] * stds[j]);
      }
    }
  } else if (id == "eggholder" || id == "holder_table") {
    bool cusp = false;
    if (id == "eggholder") {
      double a = x[0] / 2 + x[1] + 47, b = x[0] - x[1] - 47,
             A = std::sqrt(std::abs(a)), B = std::sqrt(std::abs(b));
      cusp = A < 1e-6 || B < 1e-6;
      if (!cusp) {
        double da = std::cos(A) * std::copysign(1.0, a) / (2 * A),
               db = std::cos(B) * std::copysign(1.0, b) / (2 * B);
        g[0] = -(x[1] + 47) * da / 2 - std::sin(B) - x[0] * db;
        g[1] = -std::sin(A) - (x[1] + 47) * da + x[0] * db;
      }
    } else {
      double r = std::sqrt(r2), h = 1 - r / pi, e = std::exp(std::abs(h)),
             f = std::sin(x[0]) * std::cos(x[1]) * e;
      cusp = r < 1e-6 || std::abs(h) < 1e-6 || std::abs(f) < 1e-6;
      if (!cusp) {
        double q = -std::copysign(1.0, h) / (pi * r),
               sg = -std::copysign(1.0, f);
        g[0] = sg * (std::cos(x[0]) * std::cos(x[1]) * e + f * q * x[0]);
        g[1] = sg * (-std::sin(x[0]) * std::sin(x[1]) * e + f * q * x[1]);
      }
    }
    // Symmetric central differences define a finite display/integration force
    // at cusps.
    if (cusp)
      for (int k = 0; k < d; ++k) {
        double h = 1e-5 * std::max(1.0, std::abs(x[k])), v = x[k];
        x[k] = v + h;
        double a = value(x);
        x[k] = v - h;
        double b = value(x);
        x[k] = v;
        g[k] = (a - b) / (2 * h);
      }
  }
  for (int k = 0; k < d; ++k) out[k] = static_cast<float>(g[k]);
}
void Benchmark::initial(float* x, Rng& rng) const {
  for (int k = 0; k < d; ++k)
    x[k] = float(low + (high - low) * rng.uniform01());
}
bool Benchmark::valid(const float* x) const {
  for (int k = 0; k < d; ++k)
    if (!std::isfinite(x[k]) || x[k] < low || x[k] > high) return false;
  return true;
}
void Benchmark::wrap(float* x) const {
  for (int k = 0; k < d; ++k)
    if (std::isfinite(x[k]))
      x[k] = float(low + (x[k] - low) -
                   (high - low) * std::floor((x[k] - low) / (high - low)));
}
}  // namespace fg::optimization
