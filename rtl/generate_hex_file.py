# Python script to generate a .hex file for an image
# with 128-bit words and dimensions IMG_W x IMG_H x NUM_CHANNELS.

IMG_W = 32
IMG_H = 32
NUM_CHANNELS = 1
FILENAME = "image_data.hex"

def generate_hex_file(width, height, channels, filename):
    """
    Generates a .hex file with a test pattern for multi-channel data.
    Each line contains 128 bits (16 bytes) of data.
    Uses channel-last layout: [height, width, channels]
    The pattern increments for each pixel value across all channels.
    """
    total_pixels = width * height * channels
    lines = []
    current_value = 0
    
    # Group pixels into 128-bit (16-byte) chunks
    pixels_per_line = 16
    
    print(f"Generating {width}x{height}x{channels} tensor...")
    print(f"Total pixels: {total_pixels}")
    print(f"Expected patches (stride=1): {(width-4+1)} x {(height-4+1)} = {(width-4+1)*(height-4+1)}")
    
    for i in range(0, total_pixels, pixels_per_line):
        hex_line = ""
        for j in range(pixels_per_line):
            if i + j < total_pixels:
                pixel = current_value & 0xFF
                hex_value = format(pixel, '02x')
                hex_line += hex_value
                current_value = (current_value + 1) % 256
            else:
                # Pad with zeros if we don't have enough pixels
                hex_line += "00"
        lines.append(hex_line)

    try:
        with open(filename, 'w') as f:
            for line in lines:
                f.write(line + "\n")
        print(f"Successfully generated '{filename}' with {len(lines)} lines of 128-bit data.")
        
        # Print some sample data for verification
        print("\nSample data verification:")
        print("Channel-last layout: [row, col, channel]")
        for row in range(min(3, height)):
            for col in range(min(4, width)):
                base_idx = (row * width + col) * channels
                channel_values = []
                for ch in range(channels):
                    val = (base_idx + ch) % 256
                    channel_values.append(val)
                print(f"Position ({row},{col}): channels {channel_values}")
            
    except IOError as e:
        print(f"Error writing to file '{filename}': {e}")

if __name__ == "__main__":
    generate_hex_file(IMG_W, IMG_H, NUM_CHANNELS, FILENAME)
