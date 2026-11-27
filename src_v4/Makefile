P09_ROOT ?= ${P09_ROOT}
CXX      ?= g++

# Portable flags: automatically enable AMX only when host compiler/CPU supports it, or disable via NO_AMX=1
ARCH_FLAGS := -O3 -march=native -std=c++17 -fopenmp -Wall -Wextra
HAS_AMX := $(shell grep -q amx_bf16 /proc/cpuinfo 2>/dev/null && echo 1 || echo 0)
ifeq ($(NO_AMX),1)
  ifeq ($(HAS_AMX),1)
    ARCH_FLAGS += -mno-amx-tile -mno-amx-bf16
  endif
else
  ifeq ($(HAS_AMX),1)
    ARCH_FLAGS += -mamx-tile -mamx-bf16
  endif
endif

# Probe liburing cleanly (Linux 7.0 B200 has it; Ice Lake 205 Linux 5.4 can build with NO_URING=1)
ifeq ($(NO_URING),1)
  URING_FLAGS :=
  URING_LIBS  :=
else
  HAS_URING := $(shell test -f $(P09_ROOT)/third_party/install/lib/liburing.a && echo 1 || echo 0)
  ifeq ($(HAS_URING),1)
    URING_FLAGS := -DPM_HAS_LIBURING -I$(P09_ROOT)/third_party/install/include
    URING_LIBS  := $(P09_ROOT)/third_party/install/lib/liburing.a
  else
    URING_FLAGS :=
    URING_LIBS  :=
  endif
endif

ROARING_INC := -I$(P09_ROOT)/third_party/CRoaring
ROARING_SRC := $(P09_ROOT)/third_party/CRoaring/roaring.c
ROARING_PIC := roaring_pic.o

.PHONY: all clean test

all: libpm_certvec.so certvec_cli certvec_builder_v5 certvec_bench_v4 certvec_bench_certvec5

$(ROARING_PIC): $(ROARING_SRC)
	gcc -O3 -march=native -fPIC -c $(ROARING_SRC) -o $(ROARING_PIC)

libpm_certvec.so: pm_certvec_core.cpp pm_certvec.h pm_kernels.hpp pm_s3fifo_cache.hpp pm_io_backend.hpp beta_quantiles_gen.hpp $(ROARING_PIC)
	$(CXX) $(ARCH_FLAGS) -fPIC -shared $(ROARING_INC) $(URING_FLAGS) pm_certvec_core.cpp $(ROARING_PIC) $(URING_LIBS) -o $@

certvec_cli: certvec_cli.cpp libpm_certvec.so pm_certvec.h pm_kernels.hpp pm_s3fifo_cache.hpp
	$(CXX) $(ARCH_FLAGS) $(ROARING_INC) $(URING_FLAGS) certvec_cli.cpp -L. -Wl,-rpath,'$$ORIGIN' -lpm_certvec $(ROARING_PIC) -o $@

certvec_bench_v4: certvec_bench_v4.cpp libpm_certvec.so pm_certvec.h pm_kernels.hpp
	$(CXX) $(ARCH_FLAGS) $(ROARING_INC) $(URING_FLAGS) certvec_bench_v4.cpp -L. -Wl,-rpath,'$$ORIGIN' -lpm_certvec $(ROARING_PIC) -o $@

certvec_bench_certvec5: certvec_bench_certvec5.cpp libpm_certvec.so pm_certvec.h pm_kernels.hpp
	$(CXX) $(ARCH_FLAGS) $(ROARING_INC) $(URING_FLAGS) certvec_bench_certvec5.cpp -L. -Wl,-rpath,'$$ORIGIN' -lpm_certvec $(ROARING_PIC) -o $@

certvec_builder_v5: certvec_builder_v5.cpp pm_certvec.h pm_kernels.hpp pm_s3fifo_cache.hpp $(ROARING_PIC)
	$(CXX) $(ARCH_FLAGS) $(ROARING_INC) certvec_builder_v5.cpp $(ROARING_PIC) -o $@

test: certvec_cli
	./certvec_cli --unit-test-all

clean:
	rm -f libpm_certvec.so certvec_cli certvec_builder_v5 $(ROARING_PIC)
