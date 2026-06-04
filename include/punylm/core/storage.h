#pragma once
#include <cstring>
#include <utility>
#include <vector>

#include "punylm/backend/cuda.h"
#include "punylm/core/device.h"

// A device tagged, owning, move only buffer of float

namespace punylm {
    class Storage {
    public:
        Storage() = default;

        Storage(std::size_t n, Device device) : n_(n), device_(device) {
            if (n_ == 0) return;
            if (device == Device::CPU) {
                ptr_ = new float[n_]();
            } else {
#ifdef PUNYLM_ENABLE_CUDA
                ptr_ = backend::cuda::device_malloc(n_);
                backend::cuda::memset_zero(ptr_, n_);
#else
                PUNYLM_CHECK(false, "CUDA storage requested but CUDA support is disabled!");
#endif
            }
        }

        ~Storage() { release(); }

        Storage(const Storage &) = delete;

        Storage &operator=(const Storage &) = delete;

        Storage(Storage &&other) noexcept { swap(other); }

        Storage &operator=(Storage &&other) noexcept {
            if (this != &other) {
                release();
                swap(other);
            }
            return *this;
        }

        [[nodiscard]] std::size_t size() const { return n_; }
        [[nodiscard]] Device device() const { return device_; }
        [[nodiscard]] float *data() { return ptr_; }
        [[nodiscard]] const float *data() const { return ptr_; }
        [[nodiscard]] bool empty() const { return n_ == 0; }

        void zero() {
            if (n_ == 0) return;
            if (device_ == Device::CPU) {
                std::memset(ptr_, 0, n_ * sizeof(float));
            } else {
#ifdef PUNYLM_ENABLE_CUDA
                backend::cuda::memset_zero(ptr_, n_);
#else
                PUNYLM_CHECK(false, "CUDA zero requested but CUDA support is disabled!");
#endif
            }
        }

        // Host storage transfer
        void from_host(const std::vector<float> &v) {
            PUNYLM_CHECK(v.size() == n_, "from_host size mismatch!");
            if (n_ == 0) return;
            if (device_ == Device::CPU) {
                std::memcpy(ptr_, v.data(), n_ * sizeof(float));
            } else {
#ifdef PUNYLM_ENABLE_CUDA
                backend::cuda::copy_h2d(ptr_, v.data(), n_);
#else
                PUNYLM_CHECK(false, "CUDA copy requested but CUDA support is disabled!");
#endif
            }
        }

        [[nodiscard]] std::vector<float> to_host() const {
            std::vector<float> v(n_);
            if (n_ == 0) return v;
            if (device_ == Device::CPU) {
                std::memcpy(v.data(), ptr_, n_ * sizeof(float));
            } else {
#ifdef PUNYLM_ENABLE_CUDA
                backend::cuda::copy_d2h(v.data(), ptr_, n_);
#else
                PUNYLM_CHECK(false, "CUDA copy requested but CUDA support is disabled!");
#endif
            }
            return v;
        }

    private:
        void release() {
            if (!ptr_) return;
            if (device_ == Device::CPU) {
                delete[] ptr_;
            } else {
#ifdef PUNYLM_ENABLE_CUDA
                backend::cuda::device_free(ptr_);
#else
                PUNYLM_CHECK(false, "CUDA free requested but CUDA support is disabled!");
#endif
            }
            ptr_ = nullptr;
            n_ = 0;
        }

        void swap(Storage &other) {
            std::swap(ptr_, other.ptr_);
            std::swap(n_, other.n_);
            std::swap(device_, other.device_);
        }

    private:
        float *ptr_ = nullptr;
        std::size_t n_ = 0;
        Device device_ = Device::CPU;
    };
} // namespace punylm
