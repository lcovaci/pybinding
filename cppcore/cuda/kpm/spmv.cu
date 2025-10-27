#include "compute/kpm_gpu.hpp"

#ifdef CPB_USE_CUDA

#include "cuda/traits.cuh"
#include "cuda/thrust.hpp"

#include <cusparse_v2.h>

#include <algorithm>
#include <complex>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

namespace cpb { namespace compute { namespace gpu { namespace detail {

namespace {
    template<class scalar_t>
    struct ValueConverter {
        using device_type = scalar_t;

        static device_type to_device(scalar_t const& value) {
            return value;
        }

        static scalar_t to_host(device_type const& value) {
            return value;
        }
    };

    template<>
    struct ValueConverter<std::complex<float>> {
        using device_type = cuFloatComplex;

        static device_type to_device(std::complex<float> const& value) {
            return make_cuFloatComplex(value.real(), value.imag());
        }

        static std::complex<float> to_host(device_type const& value) {
            return {cuCrealf(value), cuCimagf(value)};
        }
    };

    template<>
    struct ValueConverter<std::complex<double>> {
        using device_type = cuDoubleComplex;

        static device_type to_device(std::complex<double> const& value) {
            return make_cuDoubleComplex(value.real(), value.imag());
        }

        static std::complex<double> to_host(device_type const& value) {
            return {cuCreal(value), cuCimag(value)};
        }
    };

    template<class device_t>
    struct CusparseCaller;

    template<>
    struct CusparseCaller<float> {
        static cusparseStatus_t call(cusparseHandle_t handle, int rows, int cols, int nnz,
                                     float const* alpha, cusparseMatDescr_t descr,
                                     float const* values, int const* row_ptr, int const* col_ind,
                                     float const* x, float const* beta, float* y) {
            return cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  rows, cols, nnz, alpha, descr,
                                  values, row_ptr, col_ind, x, beta, y);
        }
    };

    template<>
    struct CusparseCaller<double> {
        static cusparseStatus_t call(cusparseHandle_t handle, int rows, int cols, int nnz,
                                     double const* alpha, cusparseMatDescr_t descr,
                                     double const* values, int const* row_ptr, int const* col_ind,
                                     double const* x, double const* beta, double* y) {
            return cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  rows, cols, nnz, alpha, descr,
                                  values, row_ptr, col_ind, x, beta, y);
        }
    };

    template<>
    struct CusparseCaller<cuFloatComplex> {
        static cusparseStatus_t call(cusparseHandle_t handle, int rows, int cols, int nnz,
                                     cuFloatComplex const* alpha, cusparseMatDescr_t descr,
                                     cuFloatComplex const* values, int const* row_ptr, int const* col_ind,
                                     cuFloatComplex const* x, cuFloatComplex const* beta, cuFloatComplex* y) {
            return cusparseCcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  rows, cols, nnz, alpha, descr,
                                  values, row_ptr, col_ind, x, beta, y);
        }
    };

    template<>
    struct CusparseCaller<cuDoubleComplex> {
        static cusparseStatus_t call(cusparseHandle_t handle, int rows, int cols, int nnz,
                                     cuDoubleComplex const* alpha, cusparseMatDescr_t descr,
                                     cuDoubleComplex const* values, int const* row_ptr, int const* col_ind,
                                     cuDoubleComplex const* x, cuDoubleComplex const* beta, cuDoubleComplex* y) {
            return cusparseZcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  rows, cols, nnz, alpha, descr,
                                  values, row_ptr, col_ind, x, beta, y);
        }
    };

    template<class scalar_t>
    inline bool check_dimensions(SparseMatrixX<scalar_t> const& matrix) {
        auto const rows = static_cast<int>(matrix.rows());
        auto const cols = static_cast<int>(matrix.cols());
        return rows >= 0 && cols >= 0;
    }
}

 template<class scalar_t>
 bool kpm_spmv_cuda(idx_t start, idx_t end, SparseMatrixX<scalar_t> const& matrix,
                    scalar_t const* x_data, scalar_t* y_data) {
    if (!check_dimensions(matrix)) {
        return false;
    }

    auto const total_rows = static_cast<int>(matrix.rows());
    auto const total_cols = static_cast<int>(matrix.cols());
    if (start < 0 || end < start || end > total_rows) {
        return false;
    }

    auto const rows = static_cast<int>(end - start);
    if (rows <= 0) {
        return true;
    }

    auto const* row_ptr_base = matrix.outerIndexPtr();
    auto const nnz_begin = static_cast<int>(row_ptr_base[start]);
    auto const nnz_end = static_cast<int>(row_ptr_base[end]);
    auto const nnz = nnz_end - nnz_begin;
    if (nnz < 0) {
        return false;
    }

    std::vector<int> row_ptr(rows + 1);
    for (auto i = 0; i <= rows; ++i) {
        row_ptr[i] = static_cast<int>(row_ptr_base[start + i] - row_ptr_base[start]);
    }

    std::vector<int> col_ind(nnz);
    auto const* matrix_col = matrix.innerIndexPtr();
    for (auto i = 0; i < nnz; ++i) {
        col_ind[i] = static_cast<int>(matrix_col[nnz_begin + i]);
    }

    using Converter = ValueConverter<scalar_t>;
    using device_t = typename Converter::device_type;

    std::vector<device_t> values(nnz);
    auto const* matrix_val = matrix.valuePtr();
    std::transform(matrix_val + nnz_begin, matrix_val + nnz_begin + nnz,
                   values.begin(), Converter::to_device);

    std::vector<device_t> x_host(total_cols);
    std::transform(x_data, x_data + total_cols, x_host.begin(), Converter::to_device);

    std::vector<device_t> y_host(rows);
    std::transform(y_data + start, y_data + end, y_host.begin(), Converter::to_device);

    thr::device_vector<int> d_row_ptr(row_ptr.begin(), row_ptr.end());
    thr::device_vector<int> d_col_ind(col_ind.begin(), col_ind.end());
    thr::device_vector<device_t> d_values(values.begin(), values.end());
    thr::device_vector<device_t> d_x(x_host.begin(), x_host.end());
    thr::device_vector<device_t> d_y(y_host.begin(), y_host.end());

    cusparseHandle_t handle = nullptr;
    if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS) {
        return false;
    }

    cusparseMatDescr_t descr = nullptr;
    auto status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroy(handle);
        return false;
    }

    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    auto alpha = Converter::to_device(scalar_t{1});
    auto beta = Converter::to_device(scalar_t{-1});

    status = CusparseCaller<device_t>::call(handle, rows, cols, nnz,
                                            &alpha, descr,
                                            thr::raw_pointer_cast(d_values.data()),
                                            thr::raw_pointer_cast(d_row_ptr.data()),
                                            thr::raw_pointer_cast(d_col_ind.data()),
                                            thr::raw_pointer_cast(d_x.data()),
                                            &beta,
                                            thr::raw_pointer_cast(d_y.data()));

    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        return false;
    }

    thr::copy(d_y.begin(), d_y.end(), y_host.begin());
    for (auto i = 0; i < rows; ++i) {
        y_data[start + i] = Converter::to_host(y_host[i]);
    }

    return true;
 }
 
 template bool kpm_spmv_cuda<float>(idx_t, idx_t, SparseMatrixX<float> const&, float const*, float*);
 template bool kpm_spmv_cuda<double>(idx_t, idx_t, SparseMatrixX<double> const&, double const*, double*);
 template bool kpm_spmv_cuda<std::complex<float>>(idx_t, idx_t, SparseMatrixX<std::complex<float>> const&, std::complex<float> const*, std::complex<float>*);
 template bool kpm_spmv_cuda<std::complex<double>>(idx_t, idx_t, SparseMatrixX<std::complex<double>> const&, std::complex<double> const*, std::complex<double>*);

}}}} // namespace cpb::compute::gpu::detail

#endif // CPB_USE_CUDA
