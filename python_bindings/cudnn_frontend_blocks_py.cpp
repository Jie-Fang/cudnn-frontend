
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cudnn_frontend;

std::vector<int64_t>
default_vector(void) {
    return {};
}

PYBIND11_MODULE(cudnn_frontend_block, m)
{
  // define all classes
}