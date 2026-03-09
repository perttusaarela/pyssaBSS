#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C" {
    void jdc(double *X, int *kpmaxit, double *w, double *eps, double *result);
}

py::array_t<double> py_jdc(
    py::array_t<double> X,
    py::array_t<int> kpmaxit,
    py::array_t<double> w,
    py::array_t<double> eps)
{
    auto bufX = X.request();
    auto bufK = kpmaxit.request();
    auto bufW = w.request();
    auto bufE = eps.request();

    int p = ((int*)bufK.ptr)[1];
    int result_size = p * p + 1;

    py::array_t<double> result(result_size);
    auto bufR = result.request();

    jdc(
        (double*)bufX.ptr,
        (int*)bufK.ptr,
        (double*)bufW.ptr,
        (double*)bufE.ptr,
        (double*)bufR.ptr
    );

    return result;
}

PYBIND11_MODULE(_core, m) {
    m.def("jdc", &py_jdc, "Run RJDC algorithm");
}