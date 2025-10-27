#include "compute/backend.hpp"

#include <atomic>

namespace cpb { namespace compute {

namespace {
    std::atomic<Backend> current_backend{Backend::CPU};
}

Backend get_backend() noexcept {
    return current_backend.load(std::memory_order_relaxed);
}

void set_backend(Backend backend) noexcept {
    current_backend.store(backend, std::memory_order_relaxed);
}

bool backend_available(Backend backend) noexcept {
    switch (backend) {
    case Backend::CPU:
        return true;
#ifdef CPB_USE_CUDA
    case Backend::CUDA:
        return true;
#endif
#ifdef CPB_USE_HIP
    case Backend::HIP:
        return true;
#endif
    default:
        return false;
    }
}

ScopedBackend::ScopedBackend(Backend backend) noexcept : previous(get_backend()) {
    set_backend(backend);
}

ScopedBackend::~ScopedBackend() noexcept {
    set_backend(previous);
}

}} // namespace cpb::compute
