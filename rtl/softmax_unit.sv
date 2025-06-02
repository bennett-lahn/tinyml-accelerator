`include "sys_types.svh"

// Softmax Acceleration Unit for TinyML CNN Classification
// Implements: softmax(z_i) = 2^(β * z_i) / Σ(2^(β * z_j)) for j=1 to N
// For 10-class classification with β = 1
// Uses fixed-point arithmetic with LUT-based 2^x approximation (more hardware-friendly than exp)
module softmax_unit #(
   parameter int NUM_CLASSES = 10
   ,parameter int OUTPUT_WIDTH = 32    // Width of output probabilities (Q1.31 format for better precision) 
   ,parameter int EXP_LUT_ADDR_WIDTH = 8 // 256 entry LUT for 2^x approximation
   ,parameter int EXP_LUT_WIDTH = 32   // Width of 2^x LUT values (32-bit for better range)
   ,parameter int BETA = 1              // Temperature parameter (fixed at 1)

    ,parameter string INIT_FILE = "../rtl/exp_lut.hex"
)(
   input  logic clk,
   input  logic reset,
   input  logic start,                 // Start softmax computation
   input  int8_t logits [NUM_CLASSES-1:0], // Input logits array (int8: -128 to +127)
   output logic signed [OUTPUT_WIDTH-1:0] probabilities [NUM_CLASSES-1:0], // Output probabilities (Q1.31)
   output logic valid,                 // Output valid signal
);

   // State machine for pipelined softmax computation
   typedef enum logic [2:0] {
      IDLE,
      FIND_MAX,
      COMPUTE_SHIFTED,     // New state: compute shifted logits
      COMPUTE_EXP,         // Modified state: compute 2^x from shifted values
      COMPUTE_SUM,
      COMPUTE_SOFTMAX,
      DONE
   } state_t;
   
   state_t current_state, next_state;
   
   // Internal registers and signals
   int8_t max_logit;
   int8_t shifted_logits [NUM_CLASSES-1:0];
   logic [EXP_LUT_WIDTH-1:0] exp_values [NUM_CLASSES-1:0];  // 32-bit 2^x values
   logic [EXP_LUT_WIDTH+4:0] exp_sum;  // Extra bits to prevent overflow during summation (36-bit)
   logic [$clog2(NUM_CLASSES)-1:0] class_counter;  // Properly sized counter (4 bits for NUM_CLASSES=10)
   logic [$clog2(NUM_CLASSES)-1:0] sum_counter;    // Properly sized counter (4 bits for NUM_CLASSES=10)
   
   // Intermediate signal for full precision shifted logit
   logic signed [8:0] temp_shifted_logit; // Can hold values from -255 to 0
   assign temp_shifted_logit = logits[class_counter] - max_logit;
   
   // 2^x LUT for hardware-friendly power-of-2 approximation
   // Covers range [-128, 127] (int8 range) with 256 entries
   // Values stored as 32-bit unsigned integers for excellent dynamic range
   // Much more hardware-friendly than exp(x) LUT
   logic [EXP_LUT_WIDTH-1:0] exp_lut [255:0];
   
   // Initialize 2^x LUT with pre-computed values
   // This LUT maps int8 values directly to scaled 2^(x/scale) values
   initial begin
   if (INIT_FILE != "") begin // Only initialize if a file is specified (using the new parameter)
      $display("softmax_unit: Initializing 2^x LUT from file: %s", INIT_FILE);
      $readmemh(INIT_FILE, exp_lut);
   end else begin
      //Default initialization if no file is provided, e.g., all zeros
      for (int i = 0; i < 256; i++) begin
         exp_lut[i] = {EXP_LUT_WIDTH{1'b0}};
      end
      $display("softmax_unit: No INIT_FILE specified, 2^x LUT not initialized from file by $readmemh.");
   end
   // Print LUT contents for verification (first few and last few entries)
   $display("=== 2^x LUT Contents (32-bit) ===");
   for (int i = 0; i < 16; i++) begin
      $display("LUT[%0d] = %h", i, exp_lut[i]);
   end
   $display("...");
   for (int i = 240; i < 256; i++) begin
      $display("LUT[%0d] = %h", i, exp_lut[i]);
   end
   $display("===============================");
   end
    
   // LUT address calculation from int8 logits
   function automatic logic [EXP_LUT_ADDR_WIDTH-1:0] logit_to_lut_addr(int8_t logit);
      logic [8:0] temp_addr;  // 9-bit temporary to handle 256 + negative value
      begin
         // Convert signed int8 to unsigned 8-bit address for LUT
         // int8 range [-128, 127] maps to LUT addresses [0, 255]
         if (logit < 0) begin
            temp_addr = 9'd256 + {1'b0, logit[7:0]};  // Convert negative to upper half
            logit_to_lut_addr = temp_addr[EXP_LUT_ADDR_WIDTH-1:0];
         end else begin
            logit_to_lut_addr = logit[EXP_LUT_ADDR_WIDTH-1:0];  // Positive values map directly
         end
      end
   endfunction
    
   // Divider for final probability computation (simplified reciprocal approximation)
   function automatic logic [OUTPUT_WIDTH-1:0] divide_by_sum(logic [EXP_LUT_WIDTH-1:0] numerator, logic [EXP_LUT_WIDTH+4:0] denominator);
      // For Q1.31 scaling: numerator (32 bits) + 31 zero bits = 63 bits total
      logic [EXP_LUT_WIDTH+30:0] scaled_num;  // 32+31-1 = 62, so [62:0] = 63 bits
      logic [EXP_LUT_WIDTH+30:0] division_result;  // Same width as scaled_num
      logic [EXP_LUT_WIDTH+30:0] extended_denominator;  // Same width for division
      begin
         // Scale numerator by 2^31 for Q1.31 output format
         scaled_num = {numerator, 31'b0};  // 32 bits + 31 bits = 63 bits total
         
         // Simple division (would use a proper divider in practice)
         if (denominator == 0) begin
            divide_by_sum = '0;
         end else begin
            // Extend denominator to match scaled_num width (63 bits)
            // denominator is 37 bits (EXP_LUT_WIDTH+4+1), so we need 63-37 = 26 extra bits
            extended_denominator = {{26{1'b0}}, denominator};
            division_result = scaled_num / extended_denominator;
            divide_by_sum = division_result[OUTPUT_WIDTH-1:0];  // Truncate to 32-bit output width
         end
      end
   endfunction
    
   // Local parameters for counter comparisons (properly sized)
   localparam logic [$clog2(NUM_CLASSES)-1:0] NUM_CLASSES_COUNTER = NUM_CLASSES[$clog2(NUM_CLASSES)-1:0];
   
   // Intermediate signals for combinational calculations
   logic [EXP_LUT_ADDR_WIDTH-1:0] current_lut_addr;
   logic [EXP_LUT_WIDTH-1:0] current_exp_value;  // 32-bit 2^x value
   logic [OUTPUT_WIDTH-1:0] current_probability;
   
   // Combinational logic for LUT address calculation
   always_comb begin
      current_lut_addr = logit_to_lut_addr(shifted_logits[class_counter]);
      current_exp_value = exp_lut[current_lut_addr];
   end
   
   // Combinational logic for probability calculation
   always_comb begin
      current_probability = divide_by_sum(exp_values[class_counter], exp_sum);
   end
    
   // State machine sequential logic
   always_ff @(posedge clk) begin
      if (reset) begin
         current_state <= IDLE;
         class_counter <= '0;
         sum_counter <= '0;
         max_logit <= -8'sd128; // Most negative int8 value
         exp_sum <= '0;
         valid <= 1'b0;
      end else begin
         current_state <= next_state;
         
         case (current_state)
               IDLE: begin
                  if (start) begin
                     class_counter <= '0;
                     sum_counter <= '0;
                     max_logit <= logits[0];
                     exp_sum <= '0;
                     valid <= 1'b0;
                  end
               end
               
               FIND_MAX: begin
                  // Find maximum logit for numerical stability
                  if (class_counter < NUM_CLASSES_COUNTER) begin
                     if (logits[class_counter] > max_logit) begin
                        max_logit <= logits[class_counter];
                     end
                     class_counter <= class_counter + 1'b1;
                  end
                  // Reset counter for next state when transitioning
                  if (class_counter == NUM_CLASSES_COUNTER - 1) begin
                     class_counter <= '0;
                  end
               end
               
               COMPUTE_SHIFTED: begin
                  // Compute shifted logits (subtract max for numerical stability)
                  if (class_counter < NUM_CLASSES_COUNTER) begin
                     // Clamp the shifted logit to the range interpretable by logit_to_lut_addr (effectively [-128, 0])
                     // Since logits[i] - max_logit <= 0, we only need to check lower bound.
                     if (temp_shifted_logit < -128) begin
                        shifted_logits[class_counter] <= -128;
                     end else begin
                        // temp_shifted_logit is already in [-128, 0], safe to cast to int8_t
                        shifted_logits[class_counter] <= int8_t'(temp_shifted_logit); 
                     end
                     class_counter <= class_counter + 1'b1;
                  end
                  // Reset counter for next state when transitioning
                  if (class_counter == NUM_CLASSES_COUNTER - 1) begin
                     class_counter <= '0;
                  end
               end
               
               COMPUTE_EXP: begin
                  // Compute 2^x using stored shifted logits
                  if (class_counter < NUM_CLASSES_COUNTER) begin
                     exp_values[class_counter] <= current_exp_value;
                     class_counter <= class_counter + 1'b1;
                  end
                  // Reset counter for next state when transitioning
                  if (class_counter == NUM_CLASSES_COUNTER - 1) begin
                     class_counter <= '0;
                     sum_counter <= '0; // Also reset sum_counter for COMPUTE_SUM state
                  end
               end
               
               COMPUTE_SUM: begin
                  // Sum all 2^x values
                  if (sum_counter < NUM_CLASSES_COUNTER) begin
                     exp_sum <= exp_sum + {{4{1'b0}}, exp_values[sum_counter]};  // Zero-extend exp_values to match exp_sum width (4 extra bits)
                     sum_counter <= sum_counter + 1'b1;
                  end
                  // Reset counter for next state when transitioning
                  if (sum_counter == NUM_CLASSES_COUNTER - 1) begin
                     class_counter <= '0;
                  end
               end
               
               COMPUTE_SOFTMAX: begin
                  // Compute final probabilities: 2^(shifted_logit) / sum
                  if (class_counter < NUM_CLASSES_COUNTER) begin
                     probabilities[class_counter] <= current_probability;
                     class_counter <= class_counter + 1'b1;
                  end
               end
               
               DONE: begin
                  valid <= 1'b1;
               end
               
               default: begin
                  // Default case to handle incomplete case coverage
                  current_state <= IDLE;
               end
         endcase
      end
   end
   
   // State machine combinational logic
   always_comb begin
      next_state = current_state;
      
      case (current_state)
         IDLE: begin
            if (start) next_state = FIND_MAX;
         end
         
         FIND_MAX: begin
            if (class_counter == NUM_CLASSES_COUNTER - 1) next_state = COMPUTE_SHIFTED;
         end
         
         COMPUTE_SHIFTED: begin
            if (class_counter == NUM_CLASSES_COUNTER - 1) next_state = COMPUTE_EXP;
         end
         
         COMPUTE_EXP: begin
            if (class_counter == NUM_CLASSES_COUNTER - 1) next_state = COMPUTE_SUM;
         end
         
         COMPUTE_SUM: begin
            if (sum_counter == NUM_CLASSES_COUNTER - 1) next_state = COMPUTE_SOFTMAX;
         end
         
         COMPUTE_SOFTMAX: begin
            if (class_counter == NUM_CLASSES_COUNTER - 1) next_state = DONE;
         end
         
         DONE: begin
            if (!start) next_state = IDLE;
         end
         
         default: next_state = IDLE;
      endcase
   end

endmodule 
