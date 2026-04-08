# FAQ

## When loading `.parquet` files, categorical columns seem to be returned as `int`, losing the information that they were categorical.

This is a known issue with parquet file support in Python.
Both existing libraries, `pyarrow` as well as `fastparquet` do not exactly reproduce original input data types when it comes to categorical columns.
See e.g. [Issue 29017](https://github.com/apache/arrow/issues/29017) and [Issue 27067](https://github.com/apache/arrow/issues/27067).

To ensure proper data type roundtrip, the module `octopus.utils` provides the functions `parquet_load()` and `parquet_save()` to store and reconstruct precise dtype information in the parquet metadata.
Files written with `parquet_save()` are expected to be readable with every parquet-compatible code.
Still, proper dtypes are only guaranteed to be reconstructed using `parquet_load()`.

For details on which dtypes are tested and supported, see [tests/infrastructure/test_file_io.py](https://github.com/emdgroup/octopus-automl/blob/main/tests/infrastructure/test_file_io.py).


## How does parallelization work in `octopus`, what are `n_cpus`, `n_workers`,  `n_assigned_cpus`?

Octopus uses a layered approach to parallelization.
Clearly, it is most efficient to distribute the work done for the individual outer splits onto individual CPUs/CPU groups.

If there are more CPUs available than outer splits to be processed, `octopus` activates inner parallelization which allows the individual tasks inside the workflow to distribute their work onto multiple processors within the CPU group assigned to the outer split.

Take for example a machine with 128 CPUs and a study with 32 outer splits.
Then, all outer splits are processed in parallel and the workflow tasks (which are always processed sequentially for every split) can parallelize onto 4 CPUs each ("inner parallelization").

The total number of CPUs to be used by `octopus` can be specified via the `n_cpus` attribute of the `OctoStudy`.
Its default value 0 uses all available CPUs.
Positive values specify the total number of CPUs to be used.
Negative values indicate `abs(n_cpus)` to leave free, e.g. -1 means use all but one CPU.
Setting `n_cpus` to 1 disables all parallel processing and runs the study sequentially.

Internally, the `ResourceConfig` class is responsible for handling these constraints. Therein, nomenclature is as follows:

* `n_cpus`, `n_cpus_user` is the user-defined number of CPUs to be used as described above.
* `available_cpus` is the absolute total number of CPUs available for `octopus` (no negative values, zero, None). Deduced from `n_cpus` and the hardware capabilities of the machine.
* `n_workers` is the number of parallel processes for the outer parallelization, i.e. the number of outer splits to be performed in parallel.
* `cpus_per_worker` is the number of CPUs available for inner parallelization, i.e. within an outer split.
* `n_assigned_cpus` is identical to `cpus_per_worker` and is being used in `octopus` internal code that should not care about whether it is running inside a dedicated worker or not. So, `n_assigned_cpus` always refers to inner parallelization.

Upon starting the `n_workers` worker processes, each of them can occupy `cpus_per_worker` CPUs without interfering with other workers.
While `octopus`-internal modules use their `n_assigned_cpus` parameter to adhere to this limit, parallelization in external code is sometimes difficult to control.
`octopus` does its best to transport the intent by

* Setting the environment variables `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,`NUMEXPR_NUM_THREADS` to `cpus_per_worker` and
* Calling `threadpoolctl.threadpool_limits(limits=n_cpus_per_worker)`

for/in every worker process.

## What is Ray and what does it have to do with parallelization?

[Ray](https://docs.ray.io/en/latest/ray-core/walkthrough.html) is a powerful distributed compute framework.
`octopus` is using it for inner and outer parallelization.

Outer parallelization (i.e. parallelization across outer splits) is done via [ray actors](https://docs.ray.io/en/latest/ray-core/actors.html#actor-guide) that spawn individual processes, while the inner parallelization is done via [ray tasks](https://docs.ray.io/en/latest/ray-core/tasks.html).

In general, you do not have to care about any of the details because for the most standard case (using multiple CPUs on a shared memory system, e.g. your desktop computer), `octopus` will take care for handling all the resource management, ray initialization, etc.

More complex setups (distributed compute, existing [ray clusters](https://docs.ray.io/en/latest/cluster/getting-started.html)), can be achieved by setting up an external ray cluster and publishing its head node address via the `RAY_ADDRESS` environment variable, e.g.

```bash
export RAY_ADDRESS=127.0.0.1:6379
```

Then, `octopus` will not initialize ray by itself but will instead check which resources are available on the cluster and make use of them.
Thus, in order to start one local node with 8 CPUs reserved for parallel processing, you can use the following command.

```bash
ray start --head --port=6379 --num-cpus=8
```

Then, this ray cluster can be used for `octopus` studies as follows:

```bash
RAY_ADDRESS=127.0.0.1:6379 python examples/wf_octo_autogluon.py
```

If you set `RAY_ADDRESS=auto`, `octopus` will try to connect to a running ray cluster, see [`ray.init`](https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html#ray-init) for details.

## Does `octopus` support distributed memory parallelization, e.g. on an HPC system?

Via [Ray](https://docs.ray.io/en/latest/ray-core/walkthrough.html), distributed compute is supported by `octopus`.
Currently, we do not test this extensively, but something like the following should work:

```bash
# start the head node
RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 ray start --head --port=6379 --num-cpus=8
# start some workers (e.g. on different distributed memory nodes)
RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 ray start --address='<HEAD_NODE_IP>:6379' --num-cpus=8
...

# run octopus
RAY_ADDRESS=<HEAD_NODE_IP>:6379 python examples/wf_octo_autogluon.py

# shutdown ray
ray stop
```

## Why don't the `TabularNNRegressor` and `TabularNNClassifier` models run with active inner parallelization?

Both models are using `pytorch` which ships its own OpenMP library that is incompatible with many OpenMP implementations (system-wide and/or provided by other packages).
This can lead to crashes or deadlock and macOS and Linux systems.
This is why we decided to restrict the number of parallel threads inside `pytorch` via

```py
torch.set_num_threads(1)
```

See https://github.com/pytorch/pytorch/issues/91547#issuecomment-1370011188 for more details.


## I am seeing runtime warnings like `(raylet) warning: `VIRTUAL_ENV=[...]/.venv` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead`

This means that the parallel workers detected issues with the python environment - likely due to a python or package version mismatch - and are creating a dedicated new virtual env.

We have seen this happening when running `octopus` like

```bash
uv run examples/basic_regression.py
```

If you instead activate the local virtual environment and use python directly, the issue does not appear any more

```bash
source .venv/bin/activate
python examples/basic_regression.py
```
