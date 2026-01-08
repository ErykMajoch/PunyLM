#include <iostream>

#include "punylm/tokeniser.hpp"

int main() {
  CharTokeniser t("data/vocab.txt");
  auto test = t.encode("Hello");
  for (auto x : test) {
    std::cout << x << " ";
  }
  std::cout << "\n";
  std::string output = t.decode(test);
  std::cout << output << "\n";

  return 0;
}