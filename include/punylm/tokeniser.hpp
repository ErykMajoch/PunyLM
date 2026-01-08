#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct CharTokeniser {
  int vocab_size = 0;
  std::vector<char> itos = {};
  std::unordered_map<char, int> stoi = {};

  explicit CharTokeniser(const std::string &vocab_path);
  std::vector<int> encode(const std::string &s) const;
  std::string decode(const std::vector<int> &tokens) const;
};
