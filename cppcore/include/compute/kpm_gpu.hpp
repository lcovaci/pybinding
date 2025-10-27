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
}

namespace gpu {

template<class scalar_t>
bool kpm_spmv(Backend backend, idx_t start, idx_t end,
              SparseMatrixX<scalar_t> const& matrix,
              VectorX<scalar_t> const& x, VectorX<scalar_t>& y) {
    switch (backend) {
    case Backend::CUDA:
    case Backend::HIP:
        detail::kpm_spmv_cpu(start, end, matrix, x, y);
        return true;
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
    case Backend::HIP:
        detail::kpm_spmv_cpu(start, end, matrix, x, y);
        return true;
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
    case Backend::HIP:
        detail::kpm_spmv_diagonal_cpu(start, end, matrix, x, y, m2, m3);
        return true;
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
    case Backend::HIP:
        detail::kpm_spmv_diagonal_cpu(start, end, matrix, x, y, m2, m3);
        return true;
    default:
        return false;
    }
}

}}} // namespace cpb::compute::gpu
