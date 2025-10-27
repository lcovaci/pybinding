#pragma once
#include "compute/backend.hpp"
#include "numeric/sparse.hpp"
#include "numeric/dense.hpp"
#include "support/simd.hpp"

namespace cpb { namespace compute {

namespace detail {
    template<class scalar_t>
    void kpm_spmv_cpu(idx_t start, idx_t end, SparseMatrixX<scalar_t> const& matrix,
                      VectorX<scalar_t> const& x, VectorX<scalar_t>& y);

    template<class scalar_t>
    void kpm_spmv_cpu(idx_t start, idx_t end, SparseMatrixX<scalar_t> const& matrix,
                      MatrixX<scalar_t> const& x, MatrixX<scalar_t>& y);

    template<class scalar_t>
    void kpm_spmv_diagonal_cpu(idx_t start, idx_t end, SparseMatrixX<scalar_t> const& matrix,
                               VectorX<scalar_t> const& x, VectorX<scalar_t>& y,
                               scalar_t& m2, scalar_t& m3);

    template<class scalar_t>
    void kpm_spmv_diagonal_cpu(idx_t start, idx_t end, SparseMatrixX<scalar_t> const& matrix,
                               MatrixX<scalar_t> const& x, MatrixX<scalar_t>& y,
                               simd::array<scalar_t>& m2, simd::array<scalar_t>& m3);

#ifdef CPB_USE_CUDA
    template<class scalar_t>
    bool kpm_spmv_cuda(idx_t start, idx_t end, SparseMatrixX<scalar_t> const& matrix,
                       scalar_t const* x_data, scalar_t* y_data);
#endif
}

namespace gpu {

template<class scalar_t>
bool kpm_spmv(Backend backend, idx_t start, idx_t end,
              SparseMatrixX<scalar_t> const& matrix,
              VectorX<scalar_t> const& x, VectorX<scalar_t>& y) {
    switch (backend) {
    case Backend::CUDA:
    {
#ifdef CPB_USE_CUDA
        if (detail::kpm_spmv_cuda(start, end, matrix, x.data(), y.data())) {
            return true;
        }
#endif
        return false;
    }
    case Backend::HIP:
        return false;
    default:
        return false;
    }
}

template<class scalar_t>
bool kpm_spmv(Backend backend, idx_t start, idx_t end,
              SparseMatrixX<scalar_t> const& matrix,
              MatrixX<scalar_t> const& x, MatrixX<scalar_t>& y) {
    switch (backend) {
    case Backend::CUDA:
    {
#ifdef CPB_USE_CUDA
        auto success = true;
        for (auto col = 0; col < x.cols(); ++col) {
            auto const* x_ptr = x.col(col).data();
            auto* y_ptr = y.col(col).data();
            if (!detail::kpm_spmv_cuda(start, end, matrix, x_ptr, y_ptr)) {
                success = false;
                break;
            }
        }
        return success;
#else
        return false;
#endif
    }
    case Backend::HIP:
        return false;
    default:
        return false;
    }
}

template<class scalar_t>
bool kpm_spmv_diagonal(Backend backend, idx_t start, idx_t end,
                       SparseMatrixX<scalar_t> const& matrix,
                       VectorX<scalar_t> const& x, VectorX<scalar_t>& y,
                       scalar_t& m2, scalar_t& m3) {
    switch (backend) {
    case Backend::CUDA:
    {
#ifdef CPB_USE_CUDA
        if (!detail::kpm_spmv_cuda(start, end, matrix, x.data(), y.data())) {
            return false;
        }
        auto const begin = static_cast<Eigen::Index>(start);
        auto const size = static_cast<Eigen::Index>(end - start);
        m2 += x.segment(begin, size).squaredNorm();
        m3 += y.segment(begin, size).dot(x.segment(begin, size));
        return true;
#else
        return false;
#endif
    }
    case Backend::HIP:
        return false;
    default:
        return false;
    }
}

template<class scalar_t>
bool kpm_spmv_diagonal(Backend backend, idx_t start, idx_t end,
                       SparseMatrixX<scalar_t> const& matrix,
                       MatrixX<scalar_t> const& x, MatrixX<scalar_t>& y,
                       simd::array<scalar_t>& m2, simd::array<scalar_t>& m3) {
    switch (backend) {
    case Backend::CUDA:
    {
#ifdef CPB_USE_CUDA
        auto const begin = static_cast<Eigen::Index>(start);
        auto const size = static_cast<Eigen::Index>(end - start);
        auto success = true;
        for (auto col = 0; col < x.cols(); ++col) {
            auto const* x_ptr = x.col(col).data();
            auto* y_ptr = y.col(col).data();
            if (!detail::kpm_spmv_cuda(start, end, matrix, x_ptr, y_ptr)) {
                success = false;
                break;
            }
            Map<VectorX<scalar_t> const> x_segment(x_ptr + begin, size);
            Map<VectorX<scalar_t>> y_segment(y_ptr + begin, size);
            m2[col] += x_segment.squaredNorm();
            m3[col] += y_segment.dot(x_segment);
        }
        return success;
#else
        return false;
#endif
    }
    case Backend::HIP:
        return false;
    default:
        return false;
    }
}

}}} // namespace cpb::compute::gpu
