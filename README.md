# Distributed Systems Assignment

This is notes for us developers, later we may just reword to make the proper submission README. Roughly written.

All code within `src/` and running yaml in `experiments`

### Run Instructions

Each config is in `configs/` simply run for example:

#### Using PyTest:

```bash
python -m pytest tests/test_full_workflow_yaml.py -s -v
```

#### Using main.py to run workers as subprocess:

```bash
python main.py .\configs\default_part_a.yaml
```

These will generate the results along with summary.json and summary.txt in the results folder under specified experiment name

### Code / gRPC Explanations:

To compile the files within `src/generated` we must use the below command, only the `src/generated/ps.proto` is manually created as per the required RPCs. Once prepared the classes and functions can be made within `src/server/service.py`

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. src/generated/ps.proto
```

### Repository Architecture

> **Note (Tentative):** The directory structure below outlines the baseline architecture for **Task A** (Parameter Server Core) and **Task B** (Asynchronous Client & Harness). We aim to adhere to this modular layout as closely as possible, which ensures clean separation of concerns and makes extending the system to future tasks straightforward without major refactoring. Some files are already there some needs to be created.

```text
parameter-server/
├── configs/
│   ├── default_part_a.yaml        # Simple single-worker sanity check config
│   └── zipf_bounded_delay.yaml    # Realistic workload config from Section 5.3
├── src/
│   ├── __init__.py
│   ├── generated/                 # Compiled protobuf/gRPC stubs
│   │   ├── __init__.py
│   │   ├── ps.proto               # Protocol Buffer definitions for Push/Pull/Tasks
│   │   ├── ps_pb2.py
│   │   └── ps_pb2_grpc.py
│   ├── server/
│   │   ├── __init__.py
│   │   ├── store.py               # In-memory parameter store & SGD update rule
│   │   ├── dep_tracker.py         # Task dependency resolver & condition variables
│   │   └── service.py             # ParameterServerServicer (gRPC service handler)
│   ├── client/
│   │   ├── __init__.py
│   │   ├── api.py                 # Asynchronous Client API (push, pull, Task handle)
│   │   └── types.py               # Range, TaskId, Task, Values data classes
│   └── harness/
│       ├── __init__.py
│       ├── config.py              # YAML config loader and validator
│       ├── workload.py            # Zipf/Dense key sampler & lognormal delays
│       ├── oracle.py              # Analytical target_diff gradient & Euclidean distance
│       ├── worker.py              # Emulated worker thread loop
│       ├── observer.py            # Dedicated background observer thread
│       ├── metrics.py             # Thread-safe CSV/JSON logger (ops, convergence)
│       └── driver.py              # Main entry point to launch cluster & orchestrate run
├── tests/
│   ├── __init__.py
│   ├── test_store.py              # Concurrent push aggregation unit tests
│   ├── test_deps.py               # Task dependency ordering tests
│   └── test_oracle_x1.py          # End-to-end mathematical convergence test (X1)
├── results/                       # Output directory for CSV and JSON files
├── main.py                        # Top-level executable entry point
├── compile_protos.sh              # Build script to generate Python stubs
├── requirements.txt               # Project dependencies
└── README.md                      # Setup and execution guide
```

### Implemented

- For week 1 from the PDF:
  - A1: Fully implemented with in-memory dense/sparse parameter storage, async push/pull gRPC servicer, and monotonic task tracking.
  - A2: Fully implemented via Client and Task abstractions providing ergonomic, non-blocking push/pull calls, context management, and wait_all().
  - A3: Partially implemented with client-side task handle chaining (deps=[...]), but server-side failure propagation and pipeline aborts on dependency failure remain unhandled.
  - A4: Partially implemented with end-to-end integration workflows (test_full_workflow.py and test_full_workflow_yaml.py), but negative unit tests for timeouts and invalid inputs are pending.

  - B1: Implemented with multi-round worker iteration loops executing pull-compute-push convergence cycles tagged by iteration clock $t$.
  - B2: Mostly implemented with Pydantic YAML parsing (load_config) and multi-process orchestration in driver.py, while Zipf-distributed access and artificial straggler delays are still missing.
  - B3-5: B3 (synthetic oracle generating reproducible $w^*$) and B4 (background ModelObserver streaming $L_2$ metrics and summary logs) are done; B5 (bounded staleness enforcement for $\tau$) is not yet implemented.

### Potential Issues / Possible Improvements in the implemented code

- Need to clarify and implement the consistency modes of sequency eventual etc... ?
- To check API error handling, may not be as comprehensive as coursework spec defines.
- ~~Worker IDs and Task IDs are generated from randint, a UUID would be more appropriate, does coursework spec only suggest or mandate numerical id or is string also okay? Should be fine for intial test, may be necessary to consider on test with many clients/iterations.~~
  > **Addressed:**: Task ID is now generated by the client instead of serrver according to the coursework pdf. using sequential locks to increase the IDs
- ~~Numpy code optimization could be explored currently just makes it work~~
- ~~A lot of locks being stored on server could this possibly be optimized or is it required~~
  > **Addressed:**: Used Single mutext lock with vectorized numpy array to avoid creation of lock per key which is not scalable
- A lot of tasks being stored on server task dictionary, possibly we should be able to clean up old ones? Depends on dependency management , or deal with this when it becomes an issue..?
