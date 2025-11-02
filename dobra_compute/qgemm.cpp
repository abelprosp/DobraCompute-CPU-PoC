
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <omp.h>

namespace py = pybind11;

// Quantized GEMM: C = (A * B) * (sa * sb), with optional dequant to float32.
// A: int8 [M,K], B: int8 [K,N]
// sa: float per-row scale for A [M] (optional -> scalar if size 1)
// sb: float per-col scale for B [N] (optional -> scalar if size 1)
py::tuple qgemm_i8i8(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> A,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> B,
    py::array_t<float, py::array::c_style | py::array::forcecast> sa,
    py::array_t<float, py::array::c_style | py::array::forcecast> sb,
    bool dequant)
{
    auto a = A.unchecked<2>();
    auto b = B.unchecked<2>();
    const int64_t M = a.shape(0);
    const int64_t K = a.shape(1);
    const int64_t Kb = b.shape(0);
    const int64_t N = b.shape(1);
    if (K != Kb) throw std::runtime_error("A.K must equal B.K");

    auto sa_u = sa.unchecked<1>();
    auto sb_u = sb.unchecked<1>();
    const bool scalar_sa = (sa.size() == 1);
    const bool scalar_sb = (sb.size() == 1);

    py::array_t<int32_t> C32({M, N});
    auto c32 = C32.mutable_unchecked<2>();

    // Basic tiled parallelization
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int64_t k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(a(i,k)) * static_cast<int32_t>(b(k,j));
            }
            c32(i,j) = acc;
        }
    }

    if (!dequant) {
        return py::make_tuple(C32, py::none());
    }

    py::array_t<float> C({M, N});
    auto cf = C.mutable_unchecked<2>();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; ++i) {
        const float sai = scalar_sa ? sa_u(0) : sa_u(i);
        for (int64_t j = 0; j < N; ++j) {
            const float sbj = scalar_sb ? sb_u(0) : sb_u(j);
            cf(i,j) = static_cast<float>(c32(i,j)) * sai * sbj;
        }
    }

    return py::make_tuple(C32, C);
}

PYBIND11_MODULE(dobra_qgemm, m) {
    m.doc() = "DobraCompute: int8 GEMM kernel with OpenMP (PoC)";
    m.def("qgemm_i8i8", &qgemm_i8i8,
          py::arg("A"), py::arg("B"), py::arg("sa"), py::arg("sb"),
          py::arg("dequant") = true,
          "Quantized int8 GEMM with optional dequant to float32");
}
