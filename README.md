# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

### 3.1 and 3.2 output
MAP

        ================================================================================
        Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, /Users/emi
        lymei/Documents/minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py (166)
        ================================================================================


        Parallel loop listing for  Function tensor_map.<locals>._map, /Users/emilymei/Documents/minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py (166)
        -----------------------------------------------------------------------------------------------|loop #ID
            def _map(                                                                                  |
                out: Storage,                                                                          |
                out_shape: Shape,                                                                      |
                out_strides: Strides,                                                                  |
                in_storage: Storage,                                                                   |
                in_shape: Shape,                                                                       |
                in_strides: Strides,                                                                   |
            ) -> None:                                                                                 |
                # check stride alignment                                                               |
                if np.array_equal(out_strides, in_strides) and np.array_equal(out_shape, in_shape):    |
                    for i in prange(len(out)):---------------------------------------------------------| #0
                        out[i] = fn(in_storage[i])                                                     |
                else:                                                                                  |
                    out_size = 1                                                                       |
                    for s in range(len(out_shape)):                                                    |
                        out_size *= out_shape[s]                                                       |
                    for i in prange(out_size):---------------------------------------------------------| #1
                        out_idx: Index = np.empty(MAX_DIMS, np.int32)                                  |
                        in_idx: Index = np.empty(MAX_DIMS, np.int32)                                   |
                        to_index(i, out_shape, out_idx)                                                |
                        o_pos = index_to_position(out_idx, out_strides)                                |
                        broadcast_index(out_idx, out_shape, in_shape, in_idx)                          |
                        i_pos = index_to_position(in_idx, in_strides)                                  |
                        out[o_pos] = fn(in_storage[i_pos])                                             |
        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Following the attempted fusion of parallel for-loops there are 2 parallel for-
        loop(s) (originating from loops labelled: #0, #1).
        --------------------------------------------------------------------------------
        ----------------------------- Before Optimisation ------------------------------
        --------------------------------------------------------------------------------
        ------------------------------ After Optimisation ------------------------------
        Parallel structure is already optimal.
        --------------------------------------------------------------------------------
        --------------------------------------------------------------------------------

        ---------------------------Loop invariant code motion---------------------------
        Allocation hoisting:
        The memory allocation derived from the instruction at /Users/emilymei/Documents/
        minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py (183) is hoisted out of
        the parallel loop labelled #1 (it will be performed before the loop is executed
        and reused inside the loop):
        Allocation:: out_idx: Index = np.empty(MAX_DIMS, np.int32)
            - numpy.empty() is used for the allocation.
        The memory allocation derived from the instruction at /Users/emilymei/Documents/
        minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py (184) is hoisted out of
        the parallel loop labelled #1 (it will be performed before the loop is executed
        and reused inside the loop):
        Allocation:: in_idx: Index = np.empty(MAX_DIMS, np.int32)
            - numpy.empty() is used for the allocation.
        None
ZIP

        ================================================================================
        Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, /Users/emi
        lymei/Documents/minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py (217)
        ================================================================================


        Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/emilymei/Documents/minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py (217)
        -------------------------------------------------------------------------|loop #ID
            def _zip(                                                            |
                out: Storage,                                                    |
                out_shape: Shape,                                                |
                out_strides: Strides,                                            |
                a_storage: Storage,                                              |
                a_shape: Shape,                                                  |
                a_strides: Strides,                                              |
                b_storage: Storage,                                              |
                b_shape: Shape,                                                  |
                b_strides: Strides,                                              |
            ) -> None:                                                           |
                # check stride alignment                                         |
                if (                                                             |
                    np.array_equal(out_strides, a_strides)                       |
                    and np.array_equal(out_strides, b_strides)                   |
                    and np.array_equal(out_shape, a_shape)                       |
                    and np.array_equal(out_shape, b_shape)                       |
                ):                                                               |
                    for i in prange(len(out)):-----------------------------------| #2
                        out[i] = fn(a_storage[i], b_storage[i])                  |
                else:                                                            |
                    out_size = 1                                                 |
                    for s in range(len(out_shape)):                              |
                        out_size *= out_shape[s]                                 |
                    for i in prange(out_size):-----------------------------------| #3
                        out_idx = np.empty(MAX_DIMS, np.int32)                   |
                        a_idx = np.empty(MAX_DIMS, np.int32)                     |
                        b_idx = np.empty(MAX_DIMS, np.int32)                     |
                        to_index(i, out_shape, out_idx)                          |
                        broadcast_index(out_idx, out_shape, a_shape, a_idx)      |
                        broadcast_index(out_idx, out_shape, b_shape, b_idx)      |
                        out_pos = index_to_position(out_idx, out_strides)        |
                        a_pos = index_to_position(a_idx, a_strides)              |
                        b_pos = index_to_position(b_idx, b_strides)              |
                        out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])    |
        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Following the attempted fusion of parallel for-loops there are 2 parallel for-
        loop(s) (originating from loops labelled: #2, #3).
        --------------------------------------------------------------------------------
        ----------------------------- Before Optimisation ------------------------------
        --------------------------------------------------------------------------------
        ------------------------------ After Optimisation ------------------------------
        Parallel structure is already optimal.
        --------------------------------------------------------------------------------
        --------------------------------------------------------------------------------

        ---------------------------Loop invariant code motion---------------------------
        Allocation hoisting:
        The memory allocation derived from the instruction at /Users/emilymei/Documents/
        minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py (242) is hoisted out of
        the parallel loop labelled #3 (it will be performed before the loop is executed
        and reused inside the loop):
        Allocation:: out_idx = np.empty(MAX_DIMS, np.int32)
            - numpy.empty() is used for the allocation.
        The memory allocation derived from the instruction at /Users/emilymei/Documents/
        minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py (243) is hoisted out of
        the parallel loop labelled #3 (it will be performed before the loop is executed
        and reused inside the loop):
        Allocation:: a_idx = np.empty(MAX_DIMS, np.int32)
            - numpy.empty() is used for the allocation.
        The memory allocation derived from the instruction at /Users/emilymei/Documents/
        minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py (244) is hoisted out of
        the parallel loop labelled #3 (it will be performed before the loop is executed
        and reused inside the loop):
        Allocation:: b_idx = np.empty(MAX_DIMS, np.int32)
            - numpy.empty() is used for the allocation.
        None
REDUCE

        ================================================================================
        Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, /Use
        rs/emilymei/Documents/minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py
        (277)
        ================================================================================


        Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/emilymei/Documents/minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py (277)
        ---------------------------------------------------------------------|loop #ID
            def _reduce(                                                     |
                out: Storage,                                                |
                out_shape: Shape,                                            |
                out_strides: Strides,                                        |
                a_storage: Storage,                                          |
                a_shape: Shape,                                              |
                a_strides: Strides,                                          |
                reduce_dim: int,                                             |
            ) -> None:                                                       |
                out_size = 1                                                 |
                for i in range(len(out_shape)):                              |
                    out_size *= out_shape[i]                                 |
                for i in prange(out_size):-----------------------------------| #4
                    out_idx = np.empty(MAX_DIMS, np.int32)                   |
                    a_idx = np.empty(MAX_DIMS, np.int32)                     |
                    to_index(i, out_shape, out_idx)                          |
                    # broadcast_index(out_idx, out_shape, a_shape, a_idx)    |
                    o_pos = index_to_position(out_idx, out_strides)          |
                    for j in range(len(out_shape)):                          |
                        a_idx[j] = out_idx[j]                                |
                    a_pos = index_to_position(a_idx, a_strides)              |
                    out[o_pos] = a_storage[a_pos]                            |
                    for s in range(1,a_shape[reduce_dim]):                   |
                        a_idx[reduce_dim] = s                                |
                        a_pos = index_to_position(a_idx, a_strides)          |
                        out[o_pos] = fn(out[o_pos], a_storage[a_pos])        |
        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Following the attempted fusion of parallel for-loops there are 1 parallel for-
        loop(s) (originating from loops labelled: #4).
        --------------------------------------------------------------------------------
        ----------------------------- Before Optimisation ------------------------------
        --------------------------------------------------------------------------------
        ------------------------------ After Optimisation ------------------------------
        Parallel structure is already optimal.
        --------------------------------------------------------------------------------
        --------------------------------------------------------------------------------

        ---------------------------Loop invariant code motion---------------------------
        Allocation hoisting:
        The memory allocation derived from the instruction at /Users/emilymei/Documents/
        minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py (290) is hoisted out of
        the parallel loop labelled #4 (it will be performed before the loop is executed
        and reused inside the loop):
        Allocation:: out_idx = np.empty(MAX_DIMS, np.int32)
            - numpy.empty() is used for the allocation.
        The memory allocation derived from the instruction at /Users/emilymei/Documents/
        minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py (291) is hoisted out of
        the parallel loop labelled #4 (it will be performed before the loop is executed
        and reused inside the loop):
        Allocation:: a_idx = np.empty(MAX_DIMS, np.int32)
            - numpy.empty() is used for the allocation.
        None
MATRIX MULTIPLY

        ================================================================================
        Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, /Users/emil
        ymei/Documents/minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py (307)
        ================================================================================


        Parallel loop listing for  Function _tensor_matrix_multiply, /Users/emilymei/Documents/minitorch_workspace/mod3-emilyysm/minitorch/fast_ops.py (307)
        ----------------------------------------------------------------------------------------|loop #ID
        def _tensor_matrix_multiply(                                                            |
            out: Storage,                                                                       |
            out_shape: Shape,                                                                   |
            out_strides: Strides,                                                               |
            a_storage: Storage,                                                                 |
            a_shape: Shape,                                                                     |
            a_strides: Strides,                                                                 |
            b_storage: Storage,                                                                 |
            b_shape: Shape,                                                                     |
            b_strides: Strides,                                                                 |
        ) -> None:                                                                              |
            """NUMBA tensor matrix multiply function.                                           |
                                                                                                |
            Should work for any tensor shapes that broadcast as long as                         |
                                                                                                |
            ```                                                                                 |
            assert a_shape[-1] == b_shape[-2]                                                   |
            ```                                                                                 |
                                                                                                |
            Optimizations:                                                                      |
                                                                                                |
            * Outer loop in parallel                                                            |
            * No index buffers or function calls                                                |
            * Inner loop should have no global writes, 1 multiply.                              |
                                                                                                |
                                                                                                |
            Args:                                                                               |
            ----                                                                                |
                out (Storage): storage for `out` tensor                                         |
                out_shape (Shape): shape for `out` tensor                                       |
                out_strides (Strides): strides for `out` tensor                                 |
                a_storage (Storage): storage for `a` tensor                                     |
                a_shape (Shape): shape for `a` tensor                                           |
                a_strides (Strides): strides for `a` tensor                                     |
                b_storage (Storage): storage for `b` tensor                                     |
                b_shape (Shape): shape for `b` tensor                                           |
                b_strides (Strides): strides for `b` tensor                                     |
                                                                                                |
            Returns:                                                                            |
            -------                                                                             |
                None : Fills in `out`                                                           |
                                                                                                |
            """                                                                                 |
            a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                              |
            b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                              |
                                                                                                |
            for n in prange(out_shape[0]):------------------------------------------------------| #5
                for i in range(out_shape[1]):                                                   |
                    for j in range(out_shape[2]):                                               |
                        sum = 0.0                                                               |
                        out_pos = (                                                             |
                            n * out_strides[0] + i * out_strides[1] + j * out_strides[2]        |
                        )                                                                       |
                        for k in range(a_shape[2]):                                             |
                            a_pos = n * a_batch_stride + i * a_strides[1] + k * a_strides[2]    |
                            b_pos = n * b_batch_stride + k * b_strides[1] + j * b_strides[2]    |
                            sum += a_storage[a_pos] * b_storage[b_pos]                          |
                        out[out_pos] = sum                                                      |
        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Following the attempted fusion of parallel for-loops there are 1 parallel for-
        loop(s) (originating from loops labelled: #5).
        --------------------------------------------------------------------------------
        ----------------------------- Before Optimisation ------------------------------
        --------------------------------------------------------------------------------
        ------------------------------ After Optimisation ------------------------------
        Parallel structure is already optimal.
        --------------------------------------------------------------------------------
        --------------------------------------------------------------------------------

        ---------------------------Loop invariant code motion---------------------------
        Allocation hoisting:
        No allocation hoisting found
        None