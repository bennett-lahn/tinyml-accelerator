# To run a specific testbench like test_systolic_tensor_array.py, simply run:
# make TEST=test_systolic_tensor_array (or any other test file)
# If you run into this warning: VPI: Couldn't find root handle..., make sure to run make clean when switching testbenches!

# For proper installation/usage of cocotb, use a python virtual environment

export COCOTB_RESOLVE_X ?= VALUE_ERROR

TOPLEVEL_LANG ?= systemverilog
VERILOG_INCLUDE_DIRS = ../rtl
VERILATOR := verilator
SIM := verilator

TEST ?= test_tensor_process_elem
TOPLEVEL ?= $(subst test_,,$(TEST))

VERILOG_SOURCES := $(shell find ../rtl -type f \( -name "*.sv" -o -name "*.v" \))
MODULE := $(TEST)
WAVES ?= 1
EXTRA_ARGS += --sv -Wall --trace --trace-structs -I

BUILD_DIR := sim_build/$(MODULE)
export PYTHONPATH := $(PYTHONPATH):$(shell pwd)

.PHONY: all sim clean

all: sim

sim:
	@echo "🏗️  Building simulation for $(TEST) with DUT $(TOPLEVEL)..."
	@# Pass COCOTB_RESOLVE_X down into the cocotb Makefile
	@COCOTB_RESOLVE_X=$(COCOTB_RESOLVE_X) \
	$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim \
	   MODULE=$(MODULE) \
	   TOPLEVEL=$(TOPLEVEL) \
	   SIM=$(SIM) \
	   VERILOG_SOURCES="$(VERILOG_SOURCES)" \
	   VERILOG_INCLUDE_DIRS="$(VERILOG_INCLUDE_DIRS)" \
	   EXTRA_ARGS="$(EXTRA_ARGS)" \
	   BUILD_DIR=$(BUILD_DIR)

clean:
	@rm -rf sim_build __pycache__ *.vcd *.log

