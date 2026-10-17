#!/usr/bin/env bash
set -e

echo "1. Downloading chess.hpp..."
curl -L -o chess.hpp https://raw.githubusercontent.com/Disservin/chess-library/master/include/chess.hpp

echo "2. Downloading ONNX Runtime..."
curl -L -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
tar -xzf ort.tgz
mv onnxruntime-linux-x64-1.17.1 onnxruntime

echo "3. Compiling C++ Engine..."
# Lưu ý: Bỏ -march=native vì Render build trên server khác
g++ -O3 -I. -Ionnxruntime/include nyra_engine.cpp -o nyra_engine -Lonnxruntime/lib -lonnxruntime -Wl,-rpath,onnxruntime/lib -lpthread

echo "4. Installing Python requirements..."
pip install -r requirements.txt

echo "Build completed successfully!"
