#!/bin/bash

cd debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
cp ./vulkan-tutorial ../vulkan-tutorial
