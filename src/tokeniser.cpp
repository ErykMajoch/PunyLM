#include "punylm/tokeniser.hpp"

CharTokeniser::CharTokeniser(const std::string &vocab_path) {
  std::ifstream in(vocab_path);
  if (!in) {
    throw std::runtime_error("Failed to open vocab file");
  }

  in >> vocab_size;
  std::string line = "";
  std::getline(in, line);
  itos.resize(vocab_size);

  for (int i = 0; i < vocab_size; i++) {
    if (!std::getline(in, line)) {
      throw std::runtime_error("Unexpected EOF in vocab");
    }
    if (line == "\\n") {
      itos[i] = '\n';
    } else if (line == "\\t") {
      itos[i] = '\t';
    } else if (line.size() == 1) {
      itos[i] = line[0];
    } else {
      throw std::runtime_error("Unsupported vocab entry: " + line);
    }

    stoi[itos[i]] = i;
  }
}

std::vector<int> CharTokeniser::encode(const std::string &s) const {
  std::vector<int> tokens = {};
  tokens.reserve(s.size());
  for (char c : s) {
    auto it = stoi.find(c);
    if (it == stoi.end()) {
      throw std::runtime_error("Unknown character in encode: " + c);
    }
    tokens.push_back(it->second);
  }
  return tokens;
}

std::string CharTokeniser::decode(const std::vector<int> &tokens) const {
  std::string result = "";
  result.reserve(tokens.size());
  for (int id : tokens) {
    if (id < 0 || id >= vocab_size) {
      throw std::runtime_error("Bad token id in decode: " + id);
    }
    result.push_back(itos[id]);
  }
  return result;
}