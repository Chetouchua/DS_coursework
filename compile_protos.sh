#!/usr/bin/env bash
set -e

# Ensure output directory exists and is a valid Python package
mkdir -p src/generated
touch src/generated/__init__.py

# Compile directly into src/generated using -Iprotos
python -m grpc_tools.protoc \
    -Iprotos \
    --python_out=src/generated \
    --grpc_python_out=src/generated \
    protos/ps.proto

# Fix relative import in the generated gRPC file
if [[ "$OSTYPE" == "darwin"* ]]; then
  sed -i '' 's/import ps_pb2 as ps__pb2/from . import ps_pb2 as ps__pb2/' src/generated/ps_pb2_grpc.py
else
  sed -i 's/import ps_pb2 as ps__pb2/from . import ps_pb2 as ps__pb2/' src/generated/ps_pb2_grpc.py
fi

echo "[+] Protobuf stubs successfully generated and patched in src/generated/"