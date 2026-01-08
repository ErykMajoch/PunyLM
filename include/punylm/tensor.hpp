#pragma once

#include <Eigen/Dense>
#include <random>

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

inline void xavier_uniform(Matrix &w, float gain = 1.0f, unsigned seed = 42) {
  int f_in = w.rows();
  int f_out = w.cols();
  float limit = gain * std::sqrt(6.0f / float(f_in + f_out));

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-limit, limit);

  for (int i = 0; i < f_in; i++) {
    for (int j = 0; j < f_out; j++) {
      w(i, j) = dist(rng);
    }
  }
}

inline void softmax(Vector &x) {
  float max_v = x.maxCoeff();
  x = (x.array() - max_v).exp();
  float sum = x.sum();
  x /= sum;
}
