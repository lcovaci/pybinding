#pragma once
namespace cpb { namespace compute {

enum class Backend {
    CPU,
    CUDA,
    HIP
};

Backend get_backend() noexcept;
void set_backend(Backend backend) noexcept;

bool backend_available(Backend backend) noexcept;

class ScopedBackend {
public:
    explicit ScopedBackend(Backend backend) noexcept;
    ~ScopedBackend() noexcept;
    ScopedBackend(ScopedBackend const&) = delete;
    ScopedBackend& operator=(ScopedBackend const&) = delete;
private:
    Backend previous;
};

}} // namespace cpb::compute
