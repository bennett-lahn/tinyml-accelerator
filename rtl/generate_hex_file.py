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
        # Format the current value as a two-digit hexadecimal string
        # (e.g., 0 -> "00", 10 -> "0A", 255 -> "FF")
        hex_value = format(current_value, '02X')
        lines.append(hex_value)

        # Increment and wrap around at 256 (for 8-bit values)
        current_value = (current_value + 1) % 256

    try:
        with open(filename, 'w') as f:
            for line in lines:
                f.write(line + "\n")
        print(f"Successfully generated '{filename}' with {total_pixels} lines.")
    except IOError as e:
        print(f"Error writing to file '{filename}': {e}")

if __name__ == "__main__":
    generate_hex_file(IMG_W, IMG_H, FILENAME)
