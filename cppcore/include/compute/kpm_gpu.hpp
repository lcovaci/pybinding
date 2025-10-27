#pragma once
#include "compute/backend.hpp"
#include "numeric/sparse.hpp"
#include "numeric/dense.hpp"
#include "support/simd.hpp"

namespace cpb { namespace compute { namespace gpu {

template<class scalar_t>
bool kpm_spmv(Backend backend, idx_t start, idx_t end,
              SparseMatrixX<scalar_t> const& matrix,
              VectorX<scalar_t> const& x, VectorX<scalar_t>& y) {
    (void)backend; (void)start; (void)end; (void)matrix; (void)x; (void)y;
    return false;
}

template<class scalar_t>
bool kpm_spmv(Backend backend, idx_t start, idx_t end,
              SparseMatrixX<scalar_t> const& matrix,
              MatrixX<scalar_t> const& x, MatrixX<scalar_t>& y) {
    (void)backend; (void)start; (void)end; (void)matrix; (void)x; (void)y;
    return false;
}

template<class scalar_t>
bool kpm_spmv_diagonal(Backend backend, idx_t start, idx_t end,
                       SparseMatrixX<scalar_t> const& matrix,
                       VectorX<scalar_t> const& x, VectorX<scalar_t>& y,
                       scalar_t& m2, scalar_t& m3) {
    (void)backend; (void)start; (void)end; (void)matrix; (void)x; (void)y; (void)m2; (void)m3;
    return false;
}

template<class scalar_t>
bool kpm_spmv_diagonal(Backend backend, idx_t start, idx_t end,
                       SparseMatrixX<scalar_t> const& matrix,
                       MatrixX<scalar_t> const& x, MatrixX<scalar_t>& y,
                       simd::array<scalar_t>& m2, simd::array<scalar_t>& m3) {
    (void)backend; (void)start; (void)end; (void)matrix; (void)x; (void)y; (void)m2; (void)m3;
    return false;
}

}}} // namespace cpb::compute::gpu
