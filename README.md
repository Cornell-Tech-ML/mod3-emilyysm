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

### 3.1 and 3.2 Output

---

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

### 3.5 Output
*Hidden layer 100, learning rate 0.05*

---

#### CPU
Simple

        Epoch  0  loss  4.957051285197779 correct 43
        Epoch  10  loss  1.8269049513763462 correct 50
        Epoch  20  loss  2.2829386318285065 correct 48
        Epoch  30  loss  1.6301260890455178 correct 50
        Epoch  40  loss  1.194582194450984 correct 50
        Epoch  50  loss  0.978731958407749 correct 49
        Epoch  60  loss  0.6611356990007096 correct 50
        Epoch  70  loss  0.09405564816298553 correct 49
        Epoch  80  loss  0.2191177411607787 correct 49
        Epoch  90  loss  0.21299611694509202 correct 50
        Epoch  100  loss  0.025574732259695934 correct 48
        Epoch  110  loss  0.8849098631236185 correct 50
        Epoch  120  loss  1.3781257815886783 correct 49
        Epoch  130  loss  0.04810332119767056 correct 49
        Epoch  140  loss  0.9661705305043882 correct 50
        Epoch  150  loss  0.5331614948027857 correct 49
        Epoch  160  loss  0.15190677814893233 correct 49
        Epoch  170  loss  0.7976766657457959 correct 49
        Epoch  180  loss  0.03497759583987351 correct 48
        Epoch  190  loss  1.089807166919584 correct 50
        Epoch  200  loss  0.9433927221322125 correct 50
        Epoch  210  loss  0.9311700787326918 correct 49
        Epoch  220  loss  0.008442321269979115 correct 50
        Epoch  230  loss  1.379270877539269 correct 50
        Epoch  240  loss  0.9717855548792078 correct 50
        Epoch  250  loss  0.11694475819070856 correct 49
        Epoch  260  loss  0.0011894002957763812 correct 49
        Epoch  270  loss  1.1101508649166185 correct 49
        Epoch  280  loss  0.16772783671910832 correct 50
        Epoch  290  loss  0.6023553953747776 correct 50
        Epoch  300  loss  0.629806671796582 correct 49
        Epoch  310  loss  0.008542902074010213 correct 49
        Epoch  320  loss  0.0009763268938835583 correct 49
        Epoch  330  loss  0.08480666020246382 correct 49
        Epoch  340  loss  0.9829927153396959 correct 50
        Epoch  350  loss  0.21392122481277046 correct 49
        Epoch  360  loss  1.2150839306140815 correct 49
        Epoch  370  loss  0.0051859192360926044 correct 49
        Epoch  380  loss  1.051394983303046 correct 49
        Epoch  390  loss  0.4298830542917272 correct 50
        Epoch  400  loss  0.11549661256670206 correct 49
        Epoch  410  loss  1.4685047266265592 correct 48
        Epoch  420  loss  0.9764941310769477 correct 50
        Epoch  430  loss  0.3809611563775292 correct 49
        Epoch  440  loss  1.836781370259555 correct 48
        Epoch  450  loss  0.09696580599934973 correct 50
        Epoch  460  loss  0.00010876299807126919 correct 49
        Epoch  470  loss  0.4022512766044446 correct 50
        Epoch  480  loss  0.37874317527742185 correct 49
        Epoch  490  loss  0.4516368772930217 correct 49

Split

        Epoch  0  loss  5.257754569275484 correct 42
        Epoch  10  loss  4.1308674998509165 correct 42
        Epoch  20  loss  2.6175188562415097 correct 47
        Epoch  30  loss  4.528667741661939 correct 44
        Epoch  40  loss  1.400640705154721 correct 42
        Epoch  50  loss  3.5520145570712383 correct 47
        Epoch  60  loss  5.317402173608395 correct 45
        Epoch  70  loss  1.1403518715954502 correct 46
        Epoch  80  loss  2.5895270877366454 correct 47
        Epoch  90  loss  1.0314131192945017 correct 48
        Epoch  100  loss  2.456568298783848 correct 48
        Epoch  110  loss  0.6165954635587036 correct 48
        Epoch  120  loss  2.5870378770369076 correct 48
        Epoch  130  loss  0.560470194067348 correct 48
        Epoch  140  loss  1.5343819285682376 correct 48
        Epoch  150  loss  0.7608563639829531 correct 48
        Epoch  160  loss  1.6239598050970057 correct 48
        Epoch  170  loss  2.6259349808466568 correct 49
        Epoch  180  loss  0.256545358169261 correct 48
        Epoch  190  loss  0.3046095784813813 correct 48
        Epoch  200  loss  0.9602684779431994 correct 49
        Epoch  210  loss  0.9393890736643543 correct 50
        Epoch  220  loss  2.4633041795459296 correct 50
        Epoch  230  loss  0.1673214766981814 correct 49
        Epoch  240  loss  0.4358881511532109 correct 50
        Epoch  250  loss  0.2640829231495421 correct 50
        Epoch  260  loss  0.4830922812690702 correct 49
        Epoch  270  loss  0.6747252299516131 correct 49
        Epoch  280  loss  0.40863893160877374 correct 50
        Epoch  290  loss  0.42151169327676624 correct 49
        Epoch  300  loss  0.1602981771616196 correct 49
        Epoch  310  loss  0.1429422843365556 correct 49
        Epoch  320  loss  0.029928784260072356 correct 49
        Epoch  330  loss  1.939532727932703 correct 50
        Epoch  340  loss  0.6865711272038847 correct 50
        Epoch  350  loss  1.3766386814793774 correct 50
        Epoch  360  loss  0.17770720963499675 correct 49
        Epoch  370  loss  0.18002842813155992 correct 50
        Epoch  380  loss  0.24996555623235084 correct 49
        Epoch  390  loss  0.18388434027269088 correct 50
        Epoch  400  loss  0.28146995429208377 correct 50
        Epoch  410  loss  0.7218626846467974 correct 49
        Epoch  420  loss  0.09349238918593104 correct 49
        Epoch  430  loss  0.852685285898018 correct 50
        Epoch  440  loss  0.21410372030248248 correct 50
        Epoch  450  loss  0.17737457694738792 correct 50
        Epoch  460  loss  0.19218630907100862 correct 50
        Epoch  470  loss  0.029437727413024913 correct 49
        Epoch  480  loss  0.51897992057418 correct 50
        Epoch  490  loss  0.33474894324435533 correct 50

XOR

        Epoch  0  loss  6.611521055507113 correct 29
        Epoch  10  loss  4.592038628667929 correct 42
        Epoch  20  loss  4.636133685815637 correct 43
        Epoch  30  loss  2.797330738580528 correct 43
        Epoch  40  loss  3.363791020261136 correct 45
        Epoch  50  loss  2.5473211570460346 correct 48
        Epoch  60  loss  2.424808344558394 correct 48
        Epoch  70  loss  3.768930387288464 correct 47
        Epoch  80  loss  1.2007898195309377 correct 47
        Epoch  90  loss  2.3728252179299067 correct 45
        Epoch  100  loss  1.4335440680945164 correct 46
        Epoch  110  loss  1.547106920712797 correct 48
        Epoch  120  loss  5.669949806865804 correct 43
        Epoch  130  loss  1.7481275805995935 correct 48
        Epoch  140  loss  2.330136756341987 correct 47
        Epoch  150  loss  1.1462001871392613 correct 47
        Epoch  160  loss  1.5857445296853532 correct 47
        Epoch  170  loss  3.6318115665302835 correct 43
        Epoch  180  loss  0.5851653366292099 correct 48
        Epoch  190  loss  2.2141539793244736 correct 48
        Epoch  200  loss  3.340279538722981 correct 46
        Epoch  210  loss  0.5807621979809481 correct 48
        Epoch  220  loss  0.2164103596232687 correct 48
        Epoch  230  loss  0.3507164114787216 correct 48
        Epoch  240  loss  1.8746991519476495 correct 48
        Epoch  250  loss  1.5567002449004634 correct 48
        Epoch  260  loss  1.5722024130280494 correct 49
        Epoch  270  loss  0.7189596201812226 correct 48
        Epoch  280  loss  0.742735967461557 correct 48
        Epoch  290  loss  0.8079441827679291 correct 49
        Epoch  300  loss  1.0051862043891158 correct 50
        Epoch  310  loss  1.326471827254607 correct 48
        Epoch  320  loss  1.3758263719827641 correct 50
        Epoch  330  loss  2.059442536844068 correct 49
        Epoch  340  loss  0.9273778978906301 correct 50
        Epoch  350  loss  1.0120967384278743 correct 49
        Epoch  360  loss  1.4065340695473159 correct 49
        Epoch  370  loss  0.7168345615818791 correct 49
        Epoch  380  loss  0.7682049753146291 correct 48
        Epoch  390  loss  0.4757705763225325 correct 49
        Epoch  400  loss  1.1136080206423218 correct 50
        Epoch  410  loss  0.6376336545952436 correct 50
        Epoch  420  loss  0.2944630542246556 correct 50
        Epoch  430  loss  1.0555423316990444 correct 50
        Epoch  440  loss  1.6989212887324943 correct 50
        Epoch  450  loss  0.954651159076189 correct 49
        Epoch  460  loss  1.412521130960482 correct 50
        Epoch  470  loss  1.6919495503614783 correct 50
        Epoch  480  loss  0.5706469163592014 correct 50
        Epoch  490  loss  0.9157539748544263 correct 50

#### GPU
Simple

        Epoch  0  loss  5.653476919386333 correct 36
        Epoch  10  loss  3.050737938368109 correct 38
        Epoch  20  loss  0.3806155814302759 correct 40
        Epoch  30  loss  1.3780694388813943 correct 43
        Epoch  40  loss  0.8399509655887549 correct 40
        Epoch  50  loss  1.741994072248817 correct 45
        Epoch  60  loss  1.3390518560086002 correct 48
        Epoch  70  loss  0.8894422042448084 correct 49
        Epoch  80  loss  2.2816275401573742 correct 47
        Epoch  90  loss  0.8911730915578913 correct 49
        Epoch  100  loss  0.3294948858576872 correct 50
        Epoch  110  loss  0.0971920038718351 correct 50
        Epoch  120  loss  0.3146662759806497 correct 50
        Epoch  130  loss  0.005904475332073777 correct 50
        Epoch  140  loss  0.005989480071373091 correct 50
        Epoch  150  loss  0.22230324196988002 correct 50
        Epoch  160  loss  0.19993243897329288 correct 50
        Epoch  170  loss  0.4171856902010167 correct 50
        Epoch  180  loss  0.004150003593263803 correct 50
        Epoch  190  loss  0.26005466920793924 correct 50
        Epoch  200  loss  0.1952392851980256 correct 50
        Epoch  210  loss  0.0013834090407408694 correct 50
        Epoch  220  loss  0.186754005555731 correct 50
        Epoch  230  loss  0.13621618551644166 correct 50
        Epoch  240  loss  0.1617897603255083 correct 50
        Epoch  250  loss  0.168428927141727 correct 50
        Epoch  260  loss  0.10562938454886159 correct 50
        Epoch  270  loss  0.012299346498863899 correct 50
        Epoch  280  loss  0.07297861889222033 correct 50
        Epoch  290  loss  0.20182798715538983 correct 50
        Epoch  300  loss  0.05150014868141445 correct 50
        Epoch  310  loss  0.07137328455677908 correct 50
        Epoch  320  loss  0.19290983506545836 correct 50
        Epoch  330  loss  0.23754939012624543 correct 50
        Epoch  340  loss  0.009091544362543164 correct 50
        Epoch  350  loss  0.0023385075690992604 correct 50
        Epoch  360  loss  0.0013994321573692536 correct 50
        Epoch  370  loss  0.08109720586899895 correct 50
        Epoch  380  loss  0.045300323304017315 correct 50
        Epoch  390  loss  0.04379473525716264 correct 50
        Epoch  400  loss  0.03535922402560058 correct 50
        Epoch  410  loss  0.11657745298292324 correct 50
        Epoch  420  loss  0.07933116630934715 correct 50
        Epoch  430  loss  0.08788168644107301 correct 50
        Epoch  440  loss  0.01347814775644333 correct 50
        Epoch  450  loss  0.004378486344325647 correct 50
        Epoch  460  loss  0.027453314343078943 correct 50
        Epoch  470  loss  0.0469728660841641 correct 50
        Epoch  480  loss  0.025244605799208848 correct 50
        Epoch  490  loss  0.024907575569013726 correct 50

Split

        Epoch  0  loss  6.321507120099194 correct 27
        Epoch  10  loss  6.169214320731644 correct 26
        Epoch  20  loss  8.028215992820298 correct 29
        Epoch  30  loss  7.4255461403943155 correct 27
        Epoch  40  loss  5.258933586386894 correct 27
        Epoch  50  loss  6.861630717777954 correct 36
        Epoch  60  loss  9.26717935253749 correct 36
        Epoch  70  loss  6.56867142042327 correct 37
        Epoch  80  loss  6.437646089164389 correct 37
        Epoch  90  loss  4.410368989270879 correct 37
        Epoch  100  loss  3.604704939778512 correct 37
        Epoch  110  loss  4.233630248173379 correct 37
        Epoch  120  loss  2.9641702625441972 correct 37
        Epoch  130  loss  3.144716848646247 correct 38
        Epoch  140  loss  2.8970187367290103 correct 38
        Epoch  150  loss  3.6438189707657243 correct 39
        Epoch  160  loss  2.977566812909003 correct 39
        Epoch  170  loss  3.3413746580265107 correct 40
        Epoch  180  loss  2.719423560336332 correct 41
        Epoch  190  loss  2.1954256110459913 correct 41
        Epoch  200  loss  2.324882447646445 correct 41
        Epoch  210  loss  1.4308627891741013 correct 42
        Epoch  220  loss  2.252271095438953 correct 42
        Epoch  230  loss  1.2731617465521257 correct 42
        Epoch  240  loss  2.050689325783307 correct 42
        Epoch  250  loss  1.8294744201274558 correct 43
        Epoch  260  loss  2.1087517195889887 correct 43
        Epoch  270  loss  0.9732273528850964 correct 43
        Epoch  280  loss  1.0483075188866802 correct 43
        Epoch  290  loss  0.7954129496935275 correct 43
        Epoch  300  loss  1.4878414174586791 correct 43
        Epoch  310  loss  1.6496169969122718 correct 43
        Epoch  320  loss  0.8596894785974695 correct 43
        Epoch  330  loss  1.4471485090577758 correct 43
        Epoch  340  loss  2.0957846721576563 correct 44
        Epoch  350  loss  0.7334637174494768 correct 43
        Epoch  360  loss  1.2728771013537556 correct 43
        Epoch  370  loss  0.6900894328431106 correct 43
        Epoch  380  loss  0.7865342839430653 correct 43
        Epoch  390  loss  0.7076183242333398 correct 44
        Epoch  400  loss  1.1064410860112308 correct 44
        Epoch  410  loss  1.087293289432907 correct 44
        Epoch  420  loss  1.1326735870451026 correct 47
        Epoch  430  loss  0.9423716308370478 correct 46
        Epoch  440  loss  0.7203292427686308 correct 45
        Epoch  450  loss  0.727645633863465 correct 47
        Epoch  460  loss  0.7527526441767032 correct 48
        Epoch  470  loss  2.0019420104909815 correct 48
        Epoch  480  loss  1.2852651048340864 correct 49
        Epoch  490  loss  1.23965380145004 correct 49

XOR

        Epoch  0  loss  5.88698168775402 correct 28
        Epoch  10  loss  6.497986318502013 correct 26
        Epoch  20  loss  6.539058332840805 correct 30
        Epoch  30  loss  5.206873113777636 correct 31
        Epoch  40  loss  5.14742372242435 correct 32
        Epoch  50  loss  5.970245461272347 correct 32
        Epoch  60  loss  5.111694863666891 correct 34
        Epoch  70  loss  3.0368481802058374 correct 34
        Epoch  80  loss  3.3223506620516057 correct 35
        Epoch  90  loss  4.370301059924375 correct 35
        Epoch  100  loss  4.335230587832149 correct 37
        Epoch  110  loss  3.3308891610479754 correct 37
        Epoch  120  loss  3.7436703689847715 correct 37
        Epoch  130  loss  2.669750642819955 correct 37
        Epoch  140  loss  1.7818257611297437 correct 38
        Epoch  150  loss  2.2474085831925996 correct 42
        Epoch  160  loss  2.608300554721781 correct 41
        Epoch  170  loss  2.2570585154401948 correct 40
        Epoch  180  loss  2.231188497186832 correct 42
        Epoch  190  loss  1.3682813715612212 correct 40
        Epoch  200  loss  1.5760652906448895 correct 41
        Epoch  210  loss  2.7360711390318455 correct 40
        Epoch  220  loss  1.9900284975961353 correct 40
        Epoch  230  loss  1.4935848250244352 correct 41
        Epoch  240  loss  1.1792202886600722 correct 40
        Epoch  250  loss  2.728970289461002 correct 41
        Epoch  260  loss  2.8485983354497635 correct 39
        Epoch  270  loss  0.7506593486959305 correct 41
        Epoch  280  loss  1.054917577073919 correct 40
        Epoch  290  loss  1.1396909345195863 correct 42
        Epoch  300  loss  1.1073056308333833 correct 40
        Epoch  310  loss  1.293583402435144 correct 43
        Epoch  320  loss  0.21697954806525896 correct 40
        Epoch  330  loss  0.7366561954761156 correct 42
        Epoch  340  loss  0.17210312991867702 correct 41
        Epoch  350  loss  1.1694026780682798 correct 42
        Epoch  360  loss  1.7126534756917322 correct 40
        Epoch  370  loss  0.5289276016774204 correct 42
        Epoch  380  loss  1.6838016238589946 correct 42
        Epoch  390  loss  0.9570503219904907 correct 42
        Epoch  400  loss  0.47165713001826565 correct 42
        Epoch  410  loss  0.8741001138142888 correct 40
        Epoch  420  loss  0.9075109290300878 correct 43
        Epoch  430  loss  1.6695705839180512 correct 43
        Epoch  440  loss  0.7690691883974383 correct 45
        Epoch  450  loss  1.4720943433749318 correct 46
        Epoch  460  loss  0.9179870197464426 correct 48
        Epoch  470  loss  0.4473856607366423 correct 48
        Epoch  480  loss  0.7555763280727794 correct 49
        Epoch  490  loss  0.3882906563963113 correct 50

---

#### Timing Comparison
*using simple dataset, learning rate 0.05*
Size = 64
* Fast: 0m16.423s
* GPU:

Size = 128
* Fast: 0m26.256s
* GPU:

Size = 256
* Fast: 0m39.256s
* GPU:

Size = 512
* Fast: 0m53.446s
* GPU:

Size = 1024
* Fast:
* GPU:
