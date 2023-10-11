#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend {
namespace python_bindings {

// pybinds for pygraph class
void
init_pygraph_submodule(py::module_ &);

// pybinds for all properties and helpers
void
init_properties(py::module_ &);

PYBIND11_MODULE(cudnn, m) {
    m.def("backend_version", &cudnnGetVersion);

    init_properties(m);
    init_pygraph_submodule(m);
}

}  // namespace python_bindings
}  // namespace cudnn_frontend