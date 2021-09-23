#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <HalideRuntime.h>

#include <filters2d.h>

static int buf_init(Py_buffer *view, halide_buffer_t *buf, halide_dimension_t *dim,
                    const char *name) {
    int i;
    int j;
    int jstart;
    int jstep;

    if (PyBuffer_IsContiguous(view, 'C')) {
        jstart = view->ndim - 1;
        jstep = -1;
    } else if (PyBuffer_IsContiguous(view, 'F')) {
        jstart = 0;
        jstep = 1;
    } else {
        PyErr_Format(PyExc_ValueError, "%s is not contiguous", name);
        return 0;
    }

    for (i = 0; i < view->ndim; i++) {
        /* An array which has a zero dimension is empty; reject these for now. */
        int d = view->shape[i];
        if (!(0 < d && d <= INT32_MAX)) {
            PyErr_Format(PyExc_ValueError, "%s has invalid shape", name);
            return 0;
        }

        /* A stride is zero if the corresponding dimension has been broadcasted. */
        /* TODO: What should we do with broadcasted arrays? */
        int s = view->strides[i];
        if (!(0 < s && s <= INT32_MAX && s % 4 == 0)) {
            PyErr_Format(PyExc_ValueError, "%s has invalid stride", name);
            return 0;
        }
    }

    for (i = 0, j = jstart; i < view->ndim; i++, j += jstep) {
        dim[j].extent = view->shape[i];
        dim[j].stride = view->strides[i] / 4;
    }

    buf->host = view->buf;
    buf->type.code = halide_type_float;
    buf->type.bits = 32;
    buf->type.lanes = 1;
    buf->dimensions = view->ndim;
    buf->dim = dim;
    return 1;
}

static PyObject *call(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *arr_obj;
    float scale;
    PyObject *out_obj;

    Py_buffer arr_view = {0};
    Py_buffer out_view = {0};

    halide_buffer_t arr_buf = {0};
    halide_buffer_t out_buf = {0};

    halide_dimension_t arr_dims[3] = {0};
    halide_dimension_t out_dims[4] = {0};

    PyThreadState *_save;
    int filter_result;
    PyObject *result = NULL;

    if (!PyArg_ParseTuple(args, "OfO", &arr_obj, &scale, &out_obj))
        goto err;

    arr_view.format = "f";
    out_view.format = "f";

    if (PyObject_GetBuffer(arr_obj, &arr_view, PyBUF_FORMAT | PyBUF_ANY_CONTIGUOUS))
        goto err;
    if (PyObject_GetBuffer(out_obj, &out_view,
                           PyBUF_FORMAT | PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE))
        goto err;

    if (!buf_init(&arr_view, &arr_buf, arr_dims, "arr"))
        goto err;
    if (!buf_init(&out_view, &out_buf, out_dims, "out"))
        goto err;

    _save = PyEval_SaveThread();
    filter_result = filters2d(&arr_buf, scale, &out_buf);
    PyEval_RestoreThread(_save);

    if (filter_result) {
        PyErr_Format(PyExc_RuntimeError, "halide error code %d", filter_result);
        goto err;
    }

    result = Py_None;

err:
    if (arr_view.obj)
        PyBuffer_Release(&arr_view);
    if (out_view.obj)
        PyBuffer_Release(&out_view);
    Py_XINCREF(result);
    return result;
}

static PyMethodDef methods[] = {{"call", call, METH_VARARGS}, {0}};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_fastfilters2", NULL, -1, methods,
};

PyMODINIT_FUNC PyInit__fastfilters2(void) { return PyModule_Create(&module); }
