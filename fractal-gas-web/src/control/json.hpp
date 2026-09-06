// Small configuration reader. Parsing is confined to scene compilation, never
// stepping.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fg::control {
struct Json {
  enum Kind { Null, Number, String, Boolean, Array, Object } kind = Null;
  double number = 0;
  std::string string;
  std::vector<Json> array;
  std::map<std::string, Json> object;
  const Json& operator[](const std::string& key) const {
    static const Json empty;
    auto it = object.find(key);
    return it == object.end() ? empty : it->second;
  }
  double num(double fallback = 0) const {
    if (kind == Null) return fallback;
    if (kind != Number) throw std::invalid_argument("Expected a number");
    return number;
  }
  bool flag(bool fallback = false) const {
    if (kind == Null) return fallback;
    if (kind != Boolean) throw std::invalid_argument("Expected a boolean");
    return number != 0;
  }
  std::string str(const std::string& fallback = "") const {
    if (kind == Null) return fallback;
    if (kind != String) throw std::invalid_argument("Expected a string");
    return string;
  }
  const std::vector<Json>& items() const {
    if (kind != Null && kind != Array)
      throw std::invalid_argument("Expected an array");
    return array;
  }
};

class JsonReader {
 public:
  explicit JsonReader(const std::string& text) : text_(text) {}
  Json read() {
    Json result = value(0);
    whitespace();
    if (pos_ != text_.size()) fail("Trailing input");
    return result;
  }

 private:
  const std::string& text_;
  size_t pos_ = 0;
  [[noreturn]] void fail(const char* reason) const {
    throw std::invalid_argument(std::string(reason) + " at JSON byte " +
                                std::to_string(pos_));
  }
  void whitespace() {
    while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\n' ||
                                   text_[pos_] == '\r' || text_[pos_] == '\t'))
      ++pos_;
  }
  bool take(char c) {
    whitespace();
    if (pos_ < text_.size() && text_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }
  void expect(char c) {
    if (!take(c)) fail("Unexpected token");
  }
  uint32_t hex4() {
    uint32_t n = 0;
    for (int i = 0; i < 4; ++i) {
      if (pos_ >= text_.size()) fail("Incomplete Unicode escape");
      char c = text_[pos_++];
      int v = c >= '0' && c <= '9'   ? c - '0'
              : c >= 'a' && c <= 'f' ? c - 'a' + 10
              : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                     : -1;
      if (v < 0) fail("Invalid Unicode escape");
      n = (n << 4) | static_cast<uint32_t>(v);
    }
    return n;
  }
  std::string quoted() {
    expect('"');
    std::string out;
    while (pos_ < text_.size()) {
      unsigned char c = text_[pos_++];
      if (c == '"') return out;
      if (c < 32) fail("Control character in string");
      if (c != '\\') {
        out += static_cast<char>(c);
        continue;
      }
      if (pos_ == text_.size()) fail("Incomplete escape");
      c = text_[pos_++];
      switch (c) {
        case '"':
        case '\\':
        case '/':
          out += static_cast<char>(c);
          break;
        case 'n':
          out += '\n';
          break;
        case 'r':
          out += '\r';
          break;
        case 't':
          out += '\t';
          break;
        case 'b':
          out += '\b';
          break;
        case 'f':
          out += '\f';
          break;
        case 'u': {
          uint32_t code = hex4();
          if (code >= 0xd800 && code <= 0xdbff) {
            if (pos_ + 2 > text_.size() || text_.substr(pos_, 2) != "\\u")
              fail("Missing low surrogate");
            pos_ += 2;
            uint32_t low = hex4();
            if (low < 0xdc00 || low > 0xdfff) fail("Invalid low surrogate");
            code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
          } else if (code >= 0xdc00 && code <= 0xdfff)
            fail("Unexpected low surrogate");
          if (code < 128)
            out += static_cast<char>(code);
          else if (code < 2048) {
            out += static_cast<char>(0xc0 | (code >> 6));
            out += static_cast<char>(0x80 | (code & 63));
          } else if (code < 65536) {
            out += static_cast<char>(0xe0 | (code >> 12));
            out += static_cast<char>(0x80 | ((code >> 6) & 63));
            out += static_cast<char>(0x80 | (code & 63));
          } else {
            out += static_cast<char>(0xf0 | (code >> 18));
            out += static_cast<char>(0x80 | ((code >> 12) & 63));
            out += static_cast<char>(0x80 | ((code >> 6) & 63));
            out += static_cast<char>(0x80 | (code & 63));
          }
          break;
        }
        default:
          fail("Invalid escape");
      }
    }
    fail("Unterminated string");
  }
  Json value(int depth) {
    if (depth > 64) fail("Configuration nesting exceeds 64");
    whitespace();
    if (pos_ == text_.size()) fail("Missing value");
    Json j;
    char c = text_[pos_];
    if (c == '"') {
      j.kind = Json::String;
      j.string = quoted();
      return j;
    }
    if (take('[')) {
      j.kind = Json::Array;
      if (take(']')) return j;
      do {
        j.array.push_back(value(depth + 1));
      } while (take(','));
      expect(']');
      return j;
    }
    if (take('{')) {
      j.kind = Json::Object;
      if (take('}')) return j;
      do {
        std::string key = quoted();
        expect(':');
        if (!j.object.emplace(key, value(depth + 1)).second)
          fail("Duplicate key");
      } while (take(','));
      expect('}');
      return j;
    }
    for (const char* literal : {"true", "false", "null"}) {
      std::string s(literal);
      if (text_.compare(pos_, s.size(), s) == 0) {
        pos_ += s.size();
        j.kind = s == "null" ? Json::Null : Json::Boolean;
        j.number = s == "true";
        return j;
      }
    }
    size_t begin = pos_;
    if (text_[pos_] == '-') ++pos_;
    if (pos_ == text_.size()) fail("Invalid number");
    auto digits = [&] {
      size_t start = pos_;
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9')
        ++pos_;
      if (start == pos_) fail("Missing digits");
    };
    if (text_[pos_] == '0')
      ++pos_;
    else
      digits();
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      digits();
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-'))
        ++pos_;
      digits();
    }
    j.kind = Json::Number;
    j.number = std::strtod(text_.substr(begin, pos_ - begin).c_str(), nullptr);
    if (!std::isfinite(j.number)) fail("Non-finite number");
    return j;
  }
};
}  // namespace fg::control
