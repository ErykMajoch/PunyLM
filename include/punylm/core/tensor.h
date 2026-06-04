#pragma once
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "punylm/core/rng.h"
#include "punylm/core/shape.h"
#include "punylm/core/storage.h"

// A Tensor is a shared_ptr to a TensorImpl

namespace punylm {
    class TensorImpl {
    public:
        TensorImpl(Shape shape, Device device, bool requires_grad) : shape_(std::move(shape)), device_(device),
                                                                     requires_grad_(requires_grad),
                                                                     data_(Storage(
                                                                         static_cast<std::size_t>(
                                                                             punylm::numel(shape_)), device)) {
        }

        Shape shape_;
        Device device_;
        bool requires_grad_ = false;

        Storage data_;
        Storage grad_;
        std::vector<std::shared_ptr<TensorImpl> > parents_;
        std::function<void()> backward_fn_;

        [[nodiscard]] int64_t numel() const { return punylm::numel(shape_); }

        void ensure_grad() {
            if (grad_.size() == 0 && requires_grad_) {
                grad_ = Storage(static_cast<std::size_t>(numel()), device_); // zeroed
            }
        }

        void zero_grad() { if (grad_.size()) grad_.zero(); }
    };

    using TensorPtr = std::shared_ptr<TensorImpl>;

    class Tensor {
    public:
        Tensor() = default;

        explicit Tensor(TensorPtr p) : impl_(std::move(p)) {
        }

        // Factories
        static Tensor zeros(const Shape &s, Device device = Device::CPU, bool requires_grad = false) {
            return Tensor(std::make_shared<TensorImpl>(s, device, requires_grad));
        }

        static Tensor from_host(const Shape &s, const std::vector<float> &v, Device device = Device::CPU,
                                bool requires_grad = false) {
            auto t = zeros(s, device, requires_grad);
            t.impl_->data_.from_host(v);
            return t;
        }

        static Tensor randn(const Shape &s, float stddev, Device device = Device::CPU, bool requires_grad = true) {
            std::vector<float> v(static_cast<std::size_t>(punylm::numel(s)));
            for (auto &x: v) {
                x = RNG::global().normal(0.0f, stddev);
            }
            return from_host(s, v, device, requires_grad);
        }

        // Accessors
        [[nodiscard]] const Shape &shape() const {
            PUNYLM_CHECK(impl_, "Tensor is undefined!");
            return impl_->shape_;
        }
        [[nodiscard]] int64_t numel() const {
            PUNYLM_CHECK(impl_, "Tensor is undefined!");
            return impl_->numel();
        }
        [[nodiscard]] Device device() const {
            PUNYLM_CHECK(impl_, "Tensor is undefined!");
            return impl_->device_;
        }
        [[nodiscard]] bool requires_grad() const {
            PUNYLM_CHECK(impl_, "Tensor is undefined!");
            return impl_->requires_grad_;
        }
        [[nodiscard]] const TensorPtr &impl() const { return impl_; }
        [[nodiscard]] bool defined() const { return static_cast<bool>(impl_); }

        [[nodiscard]] float *data() {
            PUNYLM_CHECK(impl_, "Tensor is undefined!");
            return impl_->data_.data();
        }
        [[nodiscard]] const float *data() const {
            PUNYLM_CHECK(impl_, "Tensor is undefined!");
            return impl_->data_.data();
        }

        [[nodiscard]] float *grad() {
            PUNYLM_CHECK(impl_, "Tensor is undefined!");
            impl_->ensure_grad();
            return impl_->grad_.data();
        }

        [[nodiscard]] std::vector<float> to_host() const {
            PUNYLM_CHECK(impl_, "Tensor is undefined!");
            return impl_->data_.to_host();
        }
        [[nodiscard]] std::vector<float> grad_to_host() const {
            PUNYLM_CHECK(impl_, "Tensor is undefined!");
            return impl_->grad_.to_host();
        }

        void zero_grad() {
            PUNYLM_CHECK(impl_, "Tensor is undefined!");
            impl_->zero_grad();
        }
        // Autograd entry point
        void backward() {
            PUNYLM_CHECK(impl_, "backward() called on undefined Tensor!");
            PUNYLM_CHECK(impl_->requires_grad_, "backward() called on Tensor with requires_grad=false!");

            // Post order DFS over parents
            std::vector<TensorImpl *> topo;
            std::unordered_set<TensorImpl *> seen;
            build_topo(impl_.get(), seen, topo);

            // Seed root grad with ones
            impl_->ensure_grad();
            std::vector<float> ones(static_cast<std::size_t>(numel()), 1.0f);
            impl_->grad_.from_host(ones);

            // Walk in traverse, run each node's closure
            for (auto it = topo.rbegin(); it != topo.rend(); it++) {
                if ((*it)->backward_fn_) {
                    (*it)->backward_fn_();
                }
            }
        }

            // Walk in traverse, run each node's closure
            for (auto it = topo.rbegin(); it != topo.rend(); it++) {
                if ((*it)->backward_fn_) {
                    (*it)->backward_fn_();
                }
            }
        }

    private:
        static void build_topo(TensorImpl *n, std::unordered_set<TensorImpl *> &seen, std::vector<TensorImpl *> &out) {
            if (!n || seen.count(n)) return;
            seen.insert(n);
            for (auto &p: n->parents_) {
                build_topo(p.get(), seen, out);
            }
            out.push_back(n);
        }

    private:
        TensorPtr impl_;
    };
} // namespace punylm
