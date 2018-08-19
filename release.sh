#!/bin/bash

cmake \
	-B ~/projects/vulkan-tutorial/build \
	-H ~/projects/vulkan-tutorial
make -C ~/projects/vulkan-tutorial/build
cp ~/projects/vulkan-tutorial/build/vulkan-tutorial ~/projects/vulkan-tutorial/vulkan-tutorial
