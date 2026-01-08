#include "punylm/attention.hpp"

MultiHeadSelfAttention::MultiHeadSelfAttention(int d, int h)
    : d_model(d), n_heads(h), head_dim(d / h), W_Q(d, d), W_K(d, d), W_V(d, d),
      W_O(d, d) {
  xavier_uniform(W_Q);
  xavier_uniform(W_K);
  xavier_uniform(W_V);
  xavier_uniform(W_O);
}

Matrix MultiHeadSelfAttention::forward(const Matrix &H_in) const {
  int t = H_in.rows();

  // Compute Q, K and V
  Matrix Q = H_in * W_Q;
  Matrix K = H_in * W_K;
  Matrix V = H_in * W_V;

  // Compute attention outputs
  Matrix heads_concat(t, d_model);
  float scale = 1.0f / std::sqrt(float(head_dim));

  for (int h = 0; h < n_heads; h++) {
    int start = h * head_dim;
    int end = start + head_dim;
    Matrix Q_h = Q.block(0, start, t, head_dim);
    Matrix K_h = K.block(0, start, t, head_dim);
    Matrix V_h = V.block(0, start, t, head_dim);

    Matrix scores = (Q_h * K_h.transpose()) * scale;
    for (int i = 0; i < t; i++) {
      for (int j = i + 1; j < t; j++) {
        scores(i, j) = -1e9f;
      }
    }

    for (int i = 0; i < t; i++) {
      Vector row = scores.row(i);
      softmax(row);
      scores.row(i) = row.transpose();
    }

    Matrix O_h = scores * V_h;
    heads_concat.block(0, start, t, head_dim) = O_h;
  }

  // Project heads
  Matrix H_out = heads_concat * W_O;
  return H_out;
}