#include "kpm/gpu/Compute.hpp"

namespace cpb { namespace kpm {

GpuCompute::GpuCompute(compute::Backend backend, idx_t num_threads,
                       ProgressCallback progress_callback)
    : DefaultCompute(num_threads, std::move(progress_callback)), backend(backend) {}

void GpuCompute::moments(MomentsRef m, Starter const& s, AlgorithmConfig const& ac,
                         OptimizedHamiltonian const& oh) const {
    compute::ScopedBackend guard(backend);
    DefaultCompute::moments(std::move(m), s, ac, oh);
}

CudaCompute::CudaCompute(idx_t num_threads, ProgressCallback progress_callback)
    : GpuCompute(compute::Backend::CUDA, num_threads, std::move(progress_callback)) {}

HipCompute::HipCompute(idx_t num_threads, ProgressCallback progress_callback)
    : GpuCompute(compute::Backend::HIP, num_threads, std::move(progress_callback)) {}

}} // namespace cpb::kpm
