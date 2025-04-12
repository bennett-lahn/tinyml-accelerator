#include "Vtensor_process_elem.h"
#include "verilated.h"
#include <iostream>
#include <cassert>
#include <iomanip>

vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

void toggle_clock(Vtensor_process_elem* dut) {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

void apply_inputs(Vtensor_process_elem* dut, int8_t a[4], int8_t b[4], int32_t sum_in, bool load_sum, bool reset) {
    dut->reset = reset;
    dut->load_sum = load_sum;
    dut->sum_in = sum_in;
    for (int i = 0; i < 4; ++i) {
        dut->left_in[i] = a[i];
        dut->top_in[i] = b[i];
    }
}

int32_t expected_dot(int8_t a[4], int8_t b[4]) {
    int32_t acc = 0;
    for (int i = 0; i < 4; ++i)
        acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    return acc;
}

void check_equal(const std::string& name, int32_t actual, int32_t expected) {
    if (actual == expected) {
        std::cout << "✔ " << name << " PASSED (Got: " << actual << ", Expected: " << expected << ")\n";
    } else {
        std::cerr << "✘ " << name << " FAILED (Got: " << actual << ", Expected: " << expected << ")\n";
        exit(1);
    }
}

void check_array(const std::string& name, const uint8_t actual[4], const int8_t expected[4]) {
    for (int i = 0; i < 4; ++i) {
        if (int8_t(actual[i]) != expected[i]) {
            std::cerr << "✘ " << name << " FAILED at index " << i
                      << " (Got: " << int8_t(actual[i]) << ", Expected: " << int(expected[i]) << ")\n";
            exit(1);
        }
    }
    std::cout << "✔ " << name << " PASSED\n";
}


int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vtensor_process_elem* dut = new Vtensor_process_elem;

    int8_t a[4] = {1, 2, 3, 4};
    int8_t b[4] = {5, 6, 7, 8};

    std::cout << "🚀 Starting tensor_process_elem testbench...\n";

    // Test Reset
    apply_inputs(dut, a, b, 0, false, true);
    toggle_clock(dut);
    check_equal("Reset sum_out", dut->sum_out, 0);

    // Load a custom sum (load_sum)
    apply_inputs(dut, a, b, 100, true, false);
    toggle_clock(dut);
    check_equal("Load sum_out", dut->sum_out, 100);

    // Normal MAC accumulation
    apply_inputs(dut, a, b, 0, false, false);
    toggle_clock(dut);
    int32_t expected = 100 + expected_dot(a, b);
    check_equal("MAC accumulation", dut->sum_out, expected);

    // Second accumulation with same inputs
    toggle_clock(dut);
    expected += expected_dot(a, b);
    check_equal("Second MAC accumulation", dut->sum_out, expected);

    // Check operand passthrough
    check_array("Operand right_out", dut->right_out, a);
    check_array("Operand bottom_out", dut->bottom_out, b);

    // Reset again to check state clear
    apply_inputs(dut, a, b, 0, false, true);
    toggle_clock(dut);
    check_equal("Second reset", dut->sum_out, 0);

    delete dut;
    std::cout << "✅ All tests completed successfully!\n";
    return 0;
}
