#pragma once

#include "tensor.hpp"

struct TokenEmbedding {
  int vocab_size = 0;
  int dim = 0;
  Matrix w; // [vocab_size, dim]

  TokenEmbedding(int v, int d) : vocab_size(v), dim(d), w(v, d) {
    xavier_uniform(w);
  }

  void forward(int token_id, Vector &out) const { out = w.row(token_id); }
};

struct PositionalEmbedding {
  int max_len = 0;
  int dim = 0;
  Matrix w; // [max_len, dim]

  PositionalEmbedding(int l, int d) : max_len(l), dim(d), w(l, d) {
    xavier_uniform(w);
  }

  void add_position(int pos, Vector &out) const { out += w.row(pos); }
};
