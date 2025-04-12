#include "Vsystolic_tensor_array.h"
#include "verilated.h"
#include <iostream>
#include <cassert>
#include <iomanip>

vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

const int N = 4;              // Systolic array size (NxN)
const int VECTOR_WIDTH = 4;   // PE vector width

void toggle_clock(Vsystolic_tensor_array* dut) {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

// Apply A[N][VECTOR_WIDTH] and B[N][VECTOR_WIDTH]
void apply_inputs(Vsystolic_tensor_array* dut, int8_t A[N][VECTOR_WIDTH], int8_t B[N][VECTOR_WIDTH], bool load[N][N], bool reset) {
    dut->reset = reset;
    for (int i = 0; i < N; ++i) {
        for (int v = 0; v < VECTOR_WIDTH; ++v) {
            dut->A_in[i][v] = A[i][v];
            dut->B_in[i][v] = B[i][v];
        }
        for (int j = 0; j < N; ++j) {
            dut->load_sum[i][j] = load[i][j];
        }
    }
}

void check_array(const std::string& name, const int32_t actual[N][N], const int32_t expected[N][N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (actual[i][j] != expected[i][j]) {
                std::cerr << "✘ " << name << " FAILED at [" << i << "][" << j << "] "
                          << "(Got: " << actual[i][j] << ", Expected: " << expected[i][j] << ")\n";
                exit(1);
            }
        }
    }
    std::cout << "✔ " << name << " PASSED\n";
}

int32_t expected_dot(const int8_t A[VECTOR_WIDTH], const int8_t B[VECTOR_WIDTH]) {
    int32_t result = 0;
    for (int i = 0; i < VECTOR_WIDTH; ++i) {
        result += A[i] * B[i];
    }
    return result;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vsystolic_tensor_array* dut = new Vsystolic_tensor_array;

    int8_t A[N][VECTOR_WIDTH] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    int8_t B[N][VECTOR_WIDTH] = {
        {16, 15, 14, 13},
        {12, 11, 10, 9},
        {8, 7, 6, 5},
        {4, 3, 2, 1}
    };

    bool load_sum[N][N] = {
        {true, false, false, false},
        {false, true, false, false},
        {false, false, true, false},
        {false, false, false, true}
    };

    int32_t expected_C_out[N][N] = {0};

    std::cout << "🚀 Starting systolic_tensor_array testbench...\n";

    // Test Reset
    apply_inputs(dut, A, B, load_sum, true);
    toggle_clock(dut);
    check_array("Reset C_out", dut->C_out, expected_C_out);

    // Load inputs for computation
    apply_inputs(dut, A, B, load_sum, false);
    toggle_clock(dut);

    // Expected output (matrix multiply dot product)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            expected_C_out[i][j] = expected_dot(A[i], B[j]);
        }
    }

    // Check output after one cycle
    check_array("C_out after one cycle", dut->C_out, expected_C_out);

    // Simulate a few more cycles
    for (int cycle = 0; cycle < 3; ++cycle) {
        toggle_clock(dut);
    }

    check_array("Final C_out after multiple cycles", dut->C_out, expected_C_out);

    // Reset again to check behavior
    apply_inputs(dut, A, B, load_sum, true);
    toggle_clock(dut);
    check_array("Second reset C_out", dut->C_out, expected_C_out);

    delete dut;
    std::cout << "✅ All tests completed successfully!\n";
    return 0;
}
