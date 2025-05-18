# Python script to generate a .hex file for an image
# with 8-bit words and dimensions IMG_W x IMG_H.

IMG_W = 96
IMG_H = 96
FILENAME = "image_data.hex"

def generate_hex_file(width, height, filename):
    """
    Generates a .hex file with an incrementing pattern.
    Each line contains one 8-bit hexadecimal value.
    """
    total_pixels = width * height
    lines = []
    current_value = 0

    for i in range(total_pixels):
        # Format the current value as an eight-digit hexadecimal string
        # (e.g., 0 -> "00000000", 10 -> "0000000A", 255 -> "000000FF")
        hex_value = format(current_value, '08X')
        lines.append(hex_value)

        # Increment and wrap around at 4294967296 (for 32-bit values)
        current_value = (current_value + 1) % 4294967296

    try:
        with open(filename, 'w') as f:
            for line in lines:
                f.write(line + "\n")
        print(f"Successfully generated '{filename}' with {total_pixels} lines.")
    except IOError as e:
        print(f"Error writing to file '{filename}': {e}")

if __name__ == "__main__":
    generate_hex_file(IMG_W, IMG_H, FILENAME)
