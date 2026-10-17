#!/usr/bin/env bash
set -e

echo "1. Downloading chess.hpp..."
curl -L -o chess.hpp https://raw.githubusercontent.com/Disservin/chess-library/master/include/chess.hpp

echo "1b. Downloading httplib.h..."
curl -L -o httplib.h https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h

echo "2. Downloading ONNX Runtime..."
curl -L -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
tar -xzf ort.tgz
mv onnxruntime-linux-x64-1.17.1 onnxruntime

echo "3. Compiling C++ Engine..."
g++ -O3 -I. -Ionnxruntime/include nyra_server.cpp -o nyra_server -Lonnxruntime/lib -lonnxruntime -Wl,-rpath,onnxruntime/lib -lpthread

chmod +x nyra_server

echo "4. Downloading ONNX model..."
curl -L -H "Authorization: Bearer $HF_TOKEN" -o nyra_net.onnx https://huggingface.co/huynek231/nyra-chess-weights/resolve/main/nyra_net.onnx

echo "Build completed successfully!"
