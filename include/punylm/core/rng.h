#pragma once
#include <random>

namespace punylm {
    class RNG {
    public:
        static RNG &global() {
            static RNG r;
            return r;
        }

        void seed(std::uint64_t seed) { gen_.seed(seed); }

        float normal(float mean, float stddev) {
            return std::normal_distribution<float>(mean, stddev)(gen_);
        }

        float uniform(float low, float high) {
            return std::uniform_real_distribution<float>(low, high)(gen_);
        }

        std::mt19937_64 &engine() { return gen_; }

    private:
        std::mt19937_64 gen_{0xC1CADA};
    };
} // namespace punylm
