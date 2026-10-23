# Distributed Systems Assignment

This is notes for us developers, later we may just reword to make the proper submission README. Roughly written.

All code within `src/` and running yaml in `experiments`

### Run Instructions

Each experiment is in `experiments/` simply run for example:
```bash
python src/main.py experiments/simple.yaml
```
Ideally retain this path so the data folder in project root can be used to store the target weight file.

The proper testing and validation framework is to be implemented but I have used print statements to do some tests and seen that the values do converge and the system works.

### Code / gRPC Explanations:

To compile the files within `src/grpc_generated` we must use the below command, only the `src/grpc_generated/server.proto` is manually created as per the required RPCs. Once prepared the classes and functions can be made within `src/server.py`
```bash
python -m grpc_tools.protoc -I./src --python_out=./src --pyi_out=./src --grpc_python_out=./src ./src/grpc_generated/server.proto
```

### Implemented

- For week 1 from the PDF:
    - A1: Basic Data Model and API done, with common variable vector key weight store. Sparse and dense support, push support implemented, async support done with asyncio tasks handling the behaviour
    - A2: API allows ease of use for worker, API is within client_library and can see the use in client.py it serves as an abstraction wrapper tries to keep it simple
    - A3: Basic structure available, DEPENDENCIES ARE NOT IMPLEMENTED needs to be done
    - A4: Not Implemented

    - B1: within client.py T iterations and basic sim is there, 
    - B2: main.py is the yaml parser, allows things to be configurable but the actual implementation -> things like sleep/waits, random sampling are not in place - skewed key, dense range, stragglers not there yet
    - B3-5: Not Implemented , correctness only can be seen with the debug print statements in place.

### Potential Issues / Possible Improvements in the implemented code

- Worker IDs and Task IDs are generated from randint, a UUID would be more appropriate, does coursework spec only suggest or mandate numerical id or is string also okay? Should be fine for intial test, may be necessary to consider on test with many clients/iterations.
- Numpy code optimization could be explored currently just makes it work
- A lot of locks being stored on server could this possibly be optimized or is it required
- A lot of tasks being stored on server task dictionary, possibly we should be able to clean up old ones? Depends on dependency management , or deal with this when it becomes an issue..?