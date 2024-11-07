import sys

def remove_media_omitted(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"Input file '{input_file}' not found!")
        sys.exit(1)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in lines:
            if "Media omitted" not in line:
                outfile.write(line)

    print(f"Lines containing 'Media omitted' have been removed from {input_file} and saved to {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    remove_media_omitted(input_file, output_file)
