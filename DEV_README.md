# Distributed Systems Assignment

This is notes for us developers, later we may just reword to make the proper submission README. Roughly written.

All code within `src/` and running yaml in `experiments`

### Explanations:

To compile the files within `src/grpc_generated` we must use the below command, only the `src/grpc_generated/server.proto` is manually created as per the required RPCs. Once prepared the classes and functions can be made within `src/server.py`
```bash
python -m grpc_tools.protoc -I./src --python_out=./src --pyi_out=./src --grpc_python_out=./src ./src/grpc_generated/server.proto
```

### Implemented

- For week 1 from the PDF:
    - A1: Basic Data Model and API done, with common variable vector key weight store. Sparse and dense support, push support TODO, async support done
    - A2: API allows ease of use for worker, API is within client_library and can see the use in client.py
    - A3: Basic structure available, logic TODO
    - A4: Not Implemented

    - B1: Under development?
    - B2: main.py is the yaml parser, allows things to be configurable
    - B3-5: Not Implemented