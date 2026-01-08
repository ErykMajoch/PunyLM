#pragma once

#include "layernorm.hpp"
#include "tensor.hpp"

struct MultiHeadSelfAttention {
  int d_model = 0;
  int n_heads = 0;
  int head_dim = 0;

  Matrix W_Q; // [d_model, d_model]
  Matrix W_K; // [d_model, d_model]
  Matrix W_V; // [d_model, d_model]
  Matrix W_O; // [d_model, d_model]

  MultiHeadSelfAttention(int d, int h);

  Matrix forward(const Matrix &H_in) const;
};
