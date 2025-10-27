#pragma once
#include "kpm/default/Compute.hpp"
#include "compute/backend.hpp"

#include <utility>

namespace cpb { namespace kpm {

class GpuCompute : public DefaultCompute {
public:
    using ProgressCallback = DefaultCompute::ProgressCallback;

    GpuCompute(compute::Backend backend, idx_t num_threads,
               ProgressCallback progress_callback = {});

    void moments(MomentsRef m, Starter const& s, AlgorithmConfig const& ac,
                 OptimizedHamiltonian const& oh) const override;

private:
    compute::Backend backend;
};

class CudaCompute : public GpuCompute {
public:
    CudaCompute(idx_t num_threads, ProgressCallback progress_callback = {});
};

class HipCompute : public GpuCompute {
public:
    HipCompute(idx_t num_threads, ProgressCallback progress_callback = {});
};

}} // namespace cpb::kpm
