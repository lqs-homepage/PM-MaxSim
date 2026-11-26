P09_ROOT ?= ./workspace
CXX      ?= g++
CXXFLAGS := -O3 -march=native -mamx-tile -mamx-bf16 -std=c++20 -fopenmp -Wall -Wextra
INCLUDES := -I$(P09_ROOT)/third_party/CRoaring -I$(P09_ROOT)/third_party/install/include
LIBS     := $(P09_ROOT)/third_party/CRoaring/roaring.o $(P09_ROOT)/third_party/install/lib/liburing.a -lm

TARGETS := tier1_b1_builder_v4 core_index_builder_v4 pm_engine_v4 pm_multilsm_v4

.PHONY: all clean

all: $(TARGETS)

tier1_b1_builder_v4: tier1_b1_builder_v4.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lm

core_index_builder_v4: core_index_builder_v4.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(P09_ROOT)/third_party/CRoaring/roaring.o -o $@ -lm

pm_engine_v4: pm_engine_v4.cpp beta_quantiles_d128.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(LIBS) -o $@

pm_multilsm_v4: pm_multilsm_v4.cpp beta_quantiles_d128.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(LIBS) -o $@

clean:
	rm -f $(TARGETS)
