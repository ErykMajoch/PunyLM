#pragma once

#include "tensor.hpp"

struct LayerNorm {
  int dim = 0;
  Vector gamma;
  Vector beta;
  float eps = 1e-5f;

  LayerNorm(int d, float eps_ = 1e-5f) : dim(d), gamma(d), beta(d), eps(eps_) {
    gamma.setOnes();
    beta.setZero();
  }

  void forward(Vector &x) const {
    float mean = x.mean();
    float var = (x.array() - mean).square().mean();
    float inv_std = 1.0f / std::sqrt(var + eps);
    x = gamma.array() * ((x.array() - mean) * inv_std) + beta.array();
  }
}
