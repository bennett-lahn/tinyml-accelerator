import cocotb, logging
cocotb.log.setLevel(logging.DEBUG)
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
import random

# TODO: Remove next_channel_group it doesn't do anything

@cocotb.test
async def test_TPU_Datapath(dut):
    CLOCK_PERIOD = 2
    # dut._log.setLevel(logging.DEBUG)
    # Start clock
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD, units="ns").start())


    async def tick():
        """Advance one clock cycle"""
        await RisingEdge(dut.clk)

    async def reset_dut():
        """Reset the DUT"""
        dut.reset.value = 1
        await tick()
        dut.reset.value = 0
        await tick()
    # async def apply_inputs(layer_idx, channel_idx, 
    #                        controller_pos_row, controller_pos_col,
    #                        read_bias, load_bias, reset_sta, start,
    #                        img_width, img_height, num_channels_input,
    #                        pad_top, pad_bottom, pad_left, pad_right,
    #                        start_block_extraction, next_channel_group,
    #                        next_spatial_block, start_flatten, read_logits,
    #                        softmax_start, start_dense_compute, input_valid_dense
    #                         ):
    #     """Apply inputs to the DUT"""
    #     # dut.clk.value = clk
    #     dut.layer_idx.value = layer_idx
    #     dut.channel_idx.value = channel_idx
    #     dut.controller_pos_row.value = controller_pos_row
    #     dut.controller_pos_col.value = controller_pos_col
    #     dut.read_bias.value = read_bias
    #     dut.load_bias.value = load_bias
    #     dut.reset_sta.value = reset_sta
    #     dut.start.value = start
    #     dut.img_width.value = img_width
    #     dut.img_height.value = img_height
    #     dut.num_channels.value = num_channels_input
    #     dut.pad_top.value = pad_top
    #     dut.pad_bottom.value = pad_bottom
    #     dut.pad_left.value = pad_left
    #     dut.pad_right.value = pad_right
    #     dut.start_block_extraction.value = start_block_extraction
    #     dut.next_channel_group.value = next_channel_group
    #     dut.next_spatial_block.value = next_spatial_block
    #     dut.start_flatten.value = start_flatten
    #     dut.read_logits.value = read_logits
    #     dut.softmax_start.value = softmax_start
    #     dut.start_dense_compute.value = start_dense_compute
    #     dut.input_valid_dense.value = input_valid_dense
    #     await tick()

    #     # dut.write_outputs.value = write_outputs

    async def assert_bias():
        # dut._log.info("pulsing bias")
        dut.load_bias.value = 1
        await tick()
        dut.load_bias.value = 0
        await tick()

    async def assert_incr_bias_ptr():
        # dut._log.info("pulsing incr_bias_ptr")
        dut.incr_bias_ptr.value = 1
        await tick()
        dut.incr_bias_ptr.value = 0
        await tick()

    async def assert_incr_input_ptr():
        # dut._log.info("pulsing incr_input_ptr")
        dut.incr_input_ptr.value = 1
        await tick()
        dut.incr_input_ptr.value = 0
        await tick()

    async def assert_incr_weight_ptr():
        # dut._log.info("pulsing incr_weight_ptr")
        dut.incr_weight_ptr.value = 1
        await tick()
        dut.incr_weight_ptr.value = 0
        await tick()


    async def assert_reset_sta():
        # dut._log.info("pulsing reset_sta")
        dut.reset_sta.value = 1
        await tick()
        dut.reset_sta.value = 0
        await tick()

    async def assert_done():
        # dut._log.info("pulsing done")
        dut.done.value = 1
        await tick()
        dut.done.value = 0
        await tick()
    
    async def assert_start():
        # dut._log.info("pulsing start")
        dut.start.value = 1
        await tick()
        dut.start.value = 0
        await tick()
    
    async def set_univ_buffer_config(img_width, img_height, num_channels_input, pad_top, pad_bottom, pad_left, pad_right):
        dut.img_width.value = img_width
        dut.img_height.value = img_height
        dut.num_channels_input.value = num_channels_input
        dut.pad_top.value = pad_top
        dut.pad_bottom.value = pad_bottom
        dut.pad_left.value = pad_left
        dut.pad_right.value = pad_right
        await tick()
    
    async def start_extraction():
        dut.start_block_extraction.value = 1
        await tick()
        dut.start_block_extraction.value = 0
        await tick()
    
    async def next_channel_group():
        dut.next_channel_group.value = 1
        await tick()
        dut.next_channel_group.value = 0
        await tick()
    
    async def next_spatial_block():
        dut.next_spatial_block.value = 1
        await tick()
        dut.next_spatial_block.value = 0
        await tick()

    async def start_flatten():
        dut.start_flatten.value = 1
        await tick()
        dut.start_flatten.value = 0
        await tick()
    
    async def read_logits():
        # dut._log.info("Starting logits read")
        dut.read_logits.value = 1
        for _ in range(10): # Wait for logits to be loaded
            await tick()
        dut.read_logits.value = 0
        await tick()
    
    async def softmax_start():
        # dut._log.info("Starting softmax computation")
        dut.softmax_start.value = 1
        await tick()
        dut.softmax_start.value = 0
        await tick()

    async def start_dense_compute():
        dut.start_dense_compute.value = 1
        dut.input_valid_dense.value = 1
        await tick()
        dut.start_dense_compute.value = 0
        dut.input_valid_dense.value = 0
        await tick()

    async def deassert_input_valid_dense():
        dut.input_valid_dense.value = 0
        await tick()

    async def set_input_size_dense(input_size):
        dut.input_size_dense.value = input_size
        await tick()
    
    async def set_output_size_dense(output_size):
        dut.output_size_dense.value = output_size
        await tick()
    
    async def assert_reset_datapath():
        dut.reset_datapath.value = 1
        await tick()
        dut.reset_datapath.value = 0
        await tick()

    def to_signed(value, bits):
        """Convert Python int to a signed integer of 'bits' width (2's complement)."""
        mask = (1 << bits) - 1
        val = value & mask
        if (val >> (bits - 1)) & 1:  # Check sign bit
            return val - (1 << bits)
        return val

    def q1_31_to_float(q_val):
        """Convert Q1.31 fixed-point to float."""
        return float(to_signed(q_val, 32)) / 2147483648.0  # 2^31
    
    async def run_test():

        await reset_dut()
        # dut._log.info("DUT reset complete")
        # dut._log.info("Starting test sequence Layer 0, [32x32x1]")
        # Set universal buffer config
        dut.layer_idx.value = 0
        dut.channel_idx.value = 0
        await set_univ_buffer_config(img_width=32, img_height=32, num_channels_input=1, pad_top=1, pad_bottom=2, pad_left=1, pad_right=2)
        # Start block extraction
        dut.num_columns_output.value = 16
        dut.num_channels_output.value = 8
        dut.read_bias.value = 1

        # LAYER 1
        for i in range(8):
            await tick()
            await start_extraction()
            # first output channel
            for j in range(64):
                await assert_bias()
                while dut.patches_valid.value == 0:
                    await tick()
                # dut._log.info("Patches valid, proceeding with bias load")
                await assert_start()
                # dut._log.info("Bias loaded and STA started")
                while dut.all_cols_sent.value == 0:
                    await tick()
                while dut.sta_idle.value == 0:
                    await tick()
                await assert_done()
                await tick()
                await assert_reset_sta()
                # dut._log.info("STA done, proceeding with next spatial block")
                # first 4 output pixels computed
                await next_spatial_block()
            
            dut._log.info("Ouput channel complete, resetting datapath")

            await assert_reset_datapath()

            dut.channel_idx.value = dut.channel_idx.value + 1
            await tick()
        
        dut.layer_idx.value = dut.layer_idx.value + 1
        await tick()

        dut._log.info("All channels processed, proceeding with next layer")

        dut.channel_idx.value = 0

        # Start block extraction
        dut.num_columns_output.value = 8
        dut.num_channels_output.value = 16
        await set_univ_buffer_config(img_width=16, img_height=16, num_channels_input=8, pad_top=1, pad_bottom=2, pad_left=1, pad_right=2)

        # LAYER 2 
        for i in range(16): # Output channels
            await tick()
            await assert_bias()
            await assert_start()
            await start_extraction()
            for j in range(16): # Output tiles for one output channel
                while dut.all_channels_done.value == 0: # All input channels for a tile
                    while dut.patches_valid.value == 0:
                        await tick()
                    # dut._log.info("Patches valid, proceeding with bias load")
                    await assert_start()
                    # dut._log.info("Bias loaded and STA started")
                    while dut.all_cols_sent.value == 0:
                        await tick()
                    while dut.sta_idle.value == 0:
                        await tick()
                    await assert_done()
                    await tick()
                    await assert_reset_sta()
                    # dut._log.info("STA done, proceeding with next spatial block")
                    # first 4 output pixels computed
                    await next_spatial_block()
            dut.channel_idx.value = dut.channel_idx.value + 1
            dut._log.info("Output channel processed, proceeding with next channel.")
            await assert_reset_datapath()

        dut.layer_idx.value = dut.layer_idx.value + 1
        dut.channel_idx.value = 0
        dut._log.info("All channels processed, proceeding with next layer")

        # #LAYER 3
        dut.channel_idx.value = 0

        dut.num_columns_output.value = 4
        dut.num_channels_output.value = 32
        await set_univ_buffer_config(img_width=8, img_height=8, num_channels_input=16, pad_top=1, pad_bottom=2, pad_left=1, pad_right=2)

        for i in range(32):
            await tick()
            await assert_bias()
            await assert_start()
            await start_extraction()
            for j in range(4):
                while dut.all_channels_done.value == 0:
                    while dut.patches_valid.value == 0:
                        await tick()
                    # dut._log.info("Patches valid, proceeding with bias load")
                    await assert_start()
                    # dut._log.info("Bias loaded and STA started")
                    while dut.all_cols_sent.value == 0:
                        await tick()
                    while dut.sta_idle.value == 0:
                        await tick()
                    await assert_done()
                    await tick()
                    await assert_reset_sta()
                    # dut._log.info("STA done, proceeding with next spatial block")
                    # first 4 output pixels computed
                    await next_spatial_block()

            await assert_reset_datapath()
            dut.channel_idx.value = dut.channel_idx.value + 1
            dut._log.info("Output channel processed, proceeding with next channel.")

        dut.layer_idx.value = dut.layer_idx.value + 1

        dut.channel_idx.value = 0

        # dut._log.info("All channels processed, proceeding with next layer")
        
        # LAYER 4
        dut.num_columns_output.value = 2
        dut.num_channels_output.value = 64
        await set_univ_buffer_config(img_width=4, img_height=4, num_channels_input=32, pad_top=1, pad_bottom=2, pad_left=1, pad_right=2)

        for i in range(64):
            await tick()
            await assert_bias()
            await assert_start()
            await start_extraction()
            for j in range(1):
                while dut.all_channels_done.value == 0:
                    while dut.patches_valid.value == 0:
                        await tick()
                    # dut._log.info("Patches valid, proceeding with bias load")
                    await assert_start()
                    # dut._log.info("Bias loaded and STA started")
                    while dut.all_cols_sent.value == 0:
                        await tick()
                    while dut.sta_idle.value == 0:
                        await tick()
                    await assert_done()
                    await tick()
                    await assert_reset_sta()
                    # dut._log.info("STA done, proceeding with next spatial block")
                    # first 4 output pixels computed
                    await next_spatial_block()
            await assert_reset_datapath()
            dut.channel_idx.value = i
            dut._log.info("Output channel processed, proceeding with next channel.")


        dut.layer_idx.value = dut.layer_idx.value + 1

        dut._log.info("All CONV2D processed, proceeding with flattening")
        # Flattening
        await assert_reset_datapath()
        dut.flatten_stage.value = 1

        await start_flatten()

        while dut.flatten_complete.value == 0:
            await tick()

        dut.flatten_stage.value = 0

        dut._log.info("Flattening completed, proceeding with dense layer")


        # Dense layer 1
        dut.input_size_dense.value = 255
        dut.output_size_dense.value = 63

        await start_dense_compute()

        while dut.dense_compute_completed.value == 0:
            await tick()
        dut.layer_idx.value = dut.layer_idx.value + 1
        #dense layer 2

        dut.input_size_dense.value = 63
        dut.output_size_dense.value = 9
        
        await start_dense_compute()

        while dut.dense_compute_completed.value == 0:
            await tick()
        
        dut._log.info("Dense layer completed, proceeding with logits read")

        # Read logits from dense_fc_ram (10 logits for classification)
        await read_logits()
        
        dut._log.info("Logits loaded, starting softmax computation")
        
        # Start softmax computation
        await softmax_start()
        
        # Wait for softmax computation to complete
        while dut.softmax_valid.value == 0:
            await tick()
        
        dut._log.info("Softmax computation completed! Final probabilities available.")
        
        # Optional: Read and log the final probabilities for verification
        await tick()  # One more cycle to ensure probabilities are stable
        
        dut._log.info("Test sequence completed successfully!")
        # Print final probabilities
        dut._log.info("Final probabilities:")
                # Read DUT outputs
        dut_probabilities = []
        for i in range(10):
            dut_probabilities.append(dut.probabilities[i].value.signed_integer)
        
        # Convert to float probabilities
        dut_float_probs = [q1_31_to_float(prob) for prob in dut_probabilities]
        dut._log.info(f"DUT probabilities: {dut_float_probs}")

    await run_test()
        