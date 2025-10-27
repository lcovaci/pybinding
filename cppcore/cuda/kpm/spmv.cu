#include "compute/kpm_gpu.hpp"

#ifdef CPB_USE_CUDA

#include "cuda/traits.cuh"
#include "cuda/thrust.hpp"

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <complex>
#include <cstdint>
#include <limits>
#include <mutex>

namespace cpb { namespace compute { namespace gpu { namespace detail {

namespace {
    template<class scalar_t>
    struct DeviceValue { using type = scalar_t; };

    template<>
    struct DeviceValue<std::complex<float>> { using type = cuFloatComplex; };

    template<>
    struct DeviceValue<std::complex<double>> { using type = cuDoubleComplex; };

    template<class scalar_t>
    using device_value_t = typename DeviceValue<scalar_t>::type;

    template<class device_t>
    struct ScalarOps;

    template<>
    struct ScalarOps<float> {
        static float one() { return 1.0f; }
        static float minus_one() { return -1.0f; }
    };

    template<>
    struct ScalarOps<double> {
        static double one() { return 1.0; }
        static double minus_one() { return -1.0; }
    };

    template<>
    struct ScalarOps<cuFloatComplex> {
        static cuFloatComplex one() { return make_cuFloatComplex(1.0f, 0.0f); }
        static cuFloatComplex minus_one() { return make_cuFloatComplex(-1.0f, 0.0f); }
    };

    template<>
    struct ScalarOps<cuDoubleComplex> {
        static cuDoubleComplex one() { return make_cuDoubleComplex(1.0, 0.0); }
        static cuDoubleComplex minus_one() { return make_cuDoubleComplex(-1.0, 0.0); }
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
    struct MatrixCache {
        using device_t = device_value_t<scalar_t>;

        SparseMatrixX<scalar_t> const* matrix_ptr = nullptr;
        int rows = 0;
        int cols = 0;
        int nnz = 0;

        thr::device_vector<int> row_ptr;
        thr::device_vector<int> col_ind;
        thr::device_vector<device_t> values;

        thr::device_vector<device_t> x_buffer;
        thr::device_vector<device_t> y_buffer;
        thr::device_vector<int> row_ptr_slice;

        bool ensure(SparseMatrixX<scalar_t> const& matrix) {
            auto const new_rows = static_cast<int>(matrix.rows());
            auto const new_cols = static_cast<int>(matrix.cols());
            auto const new_nnz = static_cast<int>(matrix.nonZeros());

            if (matrix_ptr == &matrix && rows == new_rows && cols == new_cols && nnz == new_nnz) {
                return true;
            }

            matrix_ptr = &matrix;
            rows = new_rows;
            cols = new_cols;
            nnz = new_nnz;

            row_ptr.resize(rows + 1);
            col_ind.resize(nnz);
            values.resize(nnz);

            auto const row_bytes = static_cast<size_t>(rows + 1) * sizeof(int);
            auto const col_bytes = static_cast<size_t>(nnz) * sizeof(int);
            auto const val_bytes = static_cast<size_t>(nnz) * sizeof(device_t);

            if (rows > 0) {
                auto status = cudaMemcpy(row_ptr.data().get(), matrix.outerIndexPtr(), row_bytes,
                                         cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    reset();
                    return false;
                }
            }

            if (nnz > 0) {
                auto status = cudaMemcpy(col_ind.data().get(), matrix.innerIndexPtr(), col_bytes,
                                         cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    reset();
                    return false;
                }

                static_assert(sizeof(device_t) == sizeof(scalar_t), "device scalar size mismatch");
                status = cudaMemcpy(values.data().get(), matrix.valuePtr(), val_bytes,
                                    cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    reset();
                    return false;
                }
            }

            return true;
        }

        void reset() {
            matrix_ptr = nullptr;
            rows = 0;
            cols = 0;
            nnz = 0;
            row_ptr.clear();
            col_ind.clear();
            values.clear();
            x_buffer.clear();
            y_buffer.clear();
            row_ptr_slice.clear();
        }
    };

    template<class scalar_t>
    MatrixCache<scalar_t>& cache_instance() {
        static MatrixCache<scalar_t> cache;
        return cache;
    }

    template<class scalar_t>
    std::mutex& cache_mutex() {
        static std::mutex mutex;
        return mutex;
    }

    class CusparseHandle {
    public:
        CusparseHandle() {
            if (cusparseCreate(&handle) == CUSPARSE_STATUS_SUCCESS) {
                cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
            } else {
                handle = nullptr;
            }
        }

        ~CusparseHandle() {
            if (handle) {
                cusparseDestroy(handle);
            }
        }

        cusparseHandle_t get() const { return handle; }

    private:
        cusparseHandle_t handle = nullptr;
    };

    class CusparseDescriptor {
    public:
        CusparseDescriptor() {
            if (cusparseCreateMatDescr(&descr) == CUSPARSE_STATUS_SUCCESS) {
                cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
                cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
            } else {
                descr = nullptr;
            }
        }

        ~CusparseDescriptor() {
            if (descr) {
                cusparseDestroyMatDescr(descr);
            }
        }

        cusparseMatDescr_t get() const { return descr; }

    private:
        cusparseMatDescr_t descr = nullptr;
    };

    cusparseHandle_t thread_cusparse_handle() {
        thread_local CusparseHandle handle;
        return handle.get();
    }

    cusparseMatDescr_t thread_cusparse_descr() {
        thread_local CusparseDescriptor descr;
        return descr.get();
    }

    struct RowOffsetShift {
        int offset;

        __host__ __device__ int operator()(int value) const { return value - offset; }
    };

    template<class scalar_t>
    bool ensure_matrix_cache(SparseMatrixX<scalar_t> const& matrix) {
        auto& cache = cache_instance<scalar_t>();
        return cache.ensure(matrix);
    }

    template<class scalar_t>
    MatrixCache<scalar_t>& get_cache() {
        return cache_instance<scalar_t>();
    }

    template<class scalar_t>
    using device_t = device_value_t<scalar_t>;

    template<class scalar_t>
    device_t<scalar_t>* values_ptr(MatrixCache<scalar_t>& cache, int offset) {
        return cache.values.data().get() + offset;
    }

    template<class scalar_t>
    int const* col_ind_ptr(MatrixCache<scalar_t>& cache, int offset) {
        return cache.col_ind.data().get() + offset;
    }

    template<class scalar_t>
    device_t<scalar_t>* x_buffer_ptr(MatrixCache<scalar_t>& cache) {
        return cache.x_buffer.data().get();
    }

    template<class scalar_t>
    device_t<scalar_t>* y_buffer_ptr(MatrixCache<scalar_t>& cache) {
        return cache.y_buffer.data().get();
    }

    template<class scalar_t>
    int* row_slice_ptr(MatrixCache<scalar_t>& cache) {
        return cache.row_ptr_slice.data().get();
    }
} // namespace

template<class scalar_t>
bool kpm_spmv_cuda(idx_t start, idx_t end, SparseMatrixX<scalar_t> const& matrix,
                   scalar_t const* x_data, scalar_t* y_data) {
    auto& mutex = cache_mutex<scalar_t>();
    std::lock_guard<std::mutex> lock(mutex);

    static_assert(sizeof(device_t<scalar_t>) == sizeof(scalar_t), "CUDA value type mismatch");

    auto const total_rows = matrix.rows();
    auto const total_cols = matrix.cols();
    if (total_rows > std::numeric_limits<int>::max() ||
        total_cols > std::numeric_limits<int>::max()) {
        return false;
    }

    if (start < 0 || end < start || end > total_rows) {
        return false;
    }

    auto const rows = static_cast<int>(end - start);
    if (rows == 0) {
        return true;
    }

    auto const nnz_begin = static_cast<int>(matrix.outerIndexPtr()[start]);
    auto const nnz_end = static_cast<int>(matrix.outerIndexPtr()[end]);
    auto const nnz = nnz_end - nnz_begin;
    if (nnz < 0) {
        return false;
    }

    if (nnz == 0) {
        for (auto row = start; row < end; ++row) {
            y_data[row] = -y_data[row];
        }
        return true;
    }

    if (!ensure_matrix_cache(matrix)) {
        return false;
    }

    auto& cache = get_cache<scalar_t>();
    auto const cols = static_cast<int>(total_cols);

    cache.x_buffer.resize(cols);
    cache.y_buffer.resize(rows);
    cache.row_ptr_slice.resize(rows + 1);

    auto const handle = thread_cusparse_handle();
    auto const descr = thread_cusparse_descr();
    if (!handle || !descr) {
        return false;
    }

    auto const bytes_x = static_cast<size_t>(cols) * sizeof(device_t<scalar_t>);
    auto const bytes_y = static_cast<size_t>(rows) * sizeof(device_t<scalar_t>);
    auto const status_x = bytes_x > 0 ? cudaMemcpy(x_buffer_ptr(cache), x_data, bytes_x,
                                                   cudaMemcpyHostToDevice)
                                      : cudaSuccess;
    auto const status_y = bytes_y > 0 ? cudaMemcpy(y_buffer_ptr(cache), y_data + start, bytes_y,
                                                   cudaMemcpyHostToDevice)
                                      : cudaSuccess;
    if (status_x != cudaSuccess || status_y != cudaSuccess) {
        return false;
    }

    thr::copy(cache.row_ptr.begin() + start, cache.row_ptr.begin() + start + rows + 1,
              cache.row_ptr_slice.begin());
    thrust::transform(cache.row_ptr_slice.begin(), cache.row_ptr_slice.end(),
                      cache.row_ptr_slice.begin(), RowOffsetShift{nnz_begin});

    auto const alpha = ScalarOps<device_t<scalar_t>>::one();
    auto const beta = ScalarOps<device_t<scalar_t>>::minus_one();

    auto const status = CusparseCaller<device_t<scalar_t>>::call(handle, rows, cols, nnz,
        &alpha, descr, values_ptr(cache, nnz_begin), row_slice_ptr(cache),
        col_ind_ptr(cache, nnz_begin), x_buffer_ptr(cache), &beta, y_buffer_ptr(cache));

    if (status != CUSPARSE_STATUS_SUCCESS) {
        return false;
    }

    auto const status_out = bytes_y > 0 ? cudaMemcpy(y_data + start, y_buffer_ptr(cache), bytes_y,
                                                     cudaMemcpyDeviceToHost)
                                        : cudaSuccess;
    return status_out == cudaSuccess;
}

template bool kpm_spmv_cuda<float>(idx_t, idx_t, SparseMatrixX<float> const&, float const*, float*);
template bool kpm_spmv_cuda<double>(idx_t, idx_t, SparseMatrixX<double> const&, double const*, double*);
template bool kpm_spmv_cuda<std::complex<float>>(idx_t, idx_t, SparseMatrixX<std::complex<float>> const&, std::complex<float> const*, std::complex<float>*);
template bool kpm_spmv_cuda<std::complex<double>>(idx_t, idx_t, SparseMatrixX<std::complex<double>> const&, std::complex<double> const*, std::complex<double>*);

}}}} // namespace cpb::compute::gpu::detail

#endif // CPB_USE_CUDA
