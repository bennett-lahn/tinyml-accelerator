# To run a specific testbench like test_systolic_tensor_array.py, simply run:
# make TEST=test_systolic_tensor_array (or any other test file)

# For proper installation/usage of cocotb, use a python virtual environment

TOPLEVEL_LANG ?= systemverilog
VERILOG_INCLUDE_DIRS = ../rtl
VERILATOR := verilator
SIM := verilator

# Default test module; override with 'make TEST=test_module_name'
TEST ?= test_tensor_process_elem

# Top-level DUT module (derived from test name by default)
TOPLEVEL ?= $(subst test_,,$(TEST))

# Source files for the DUT (all files in ~/rtl)
VERILOG_SOURCES = ../rtl/sys_types.sv \
                  $(shell find ../rtl -type f \( -iname "*.sv" -o -iname "*.v" \) ! -iname "sys_types.sv")

MODULE := $(TEST)
WAVES ?= 1
EXTRA_ARGS += -sv -Wall --trace

# Output directory for compiled sim
BUILD_DIR := sim_build/$(MODULE)

export PYTHONPATH := $(PYTHONPATH):$(shell pwd)

# Default rule
all: sim

sim:
	@echo "🏗️  Building simulation for $(TEST) with DUT $(TOPLEVEL)..."
	make -f $(shell cocotb-config --makefiles)/Makefile.sim \
		MODULE=$(MODULE) \
		TOPLEVEL=$(TOPLEVEL) \
		SIM=$(SIM) \
		VERILOG_SOURCES="$(VERILOG_SOURCES)" \
		VERILOG_INCLUDE_DIRS="$(VERILOG_INCLUDE_DIRS)" \
		EXTRA_ARGS="$(EXTRA_ARGS)" \
		BUILD_DIR=$(BUILD_DIR)

clean:
	rm -rf sim_build __pycache__ *.vcd *.log

.PHONY: all sim clean
