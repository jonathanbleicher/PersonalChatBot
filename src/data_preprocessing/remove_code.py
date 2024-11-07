import sys
import re

def should_skip_line(line):
    # Function to determine if a line should cause the message to be skipped
    code_patterns = [
        # Common code patterns (C, C++, Python, etc.)
        r'//', r'#include', r'\bmodule\b', r'\bint\b', r'\bchar\b', r'\breturn\b', r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b', r'\bvoid\b',
        r'\bimport\b', r'\bfrom\b', r'\bdef\b', r'\bclass\b', r'\bprint\b', r'\breturn\b', r'\bnp\.', r'\bscipy\.sparse\b', r'\brange\b', r'\bcsr_matrix\b',

        # Comments and metadata
        r'^#', r'//',  # Python or Shell-style comments
        r'^\s*#',      # Lines that start with a '#' and possibly whitespace
        r'^\s*\/\/',   # Lines that start with '//' and possibly whitespace
        
        # LaTeX code patterns
        r'\\documentclass', r'\\usepackage', r'\\begin\{document\}', r'\\end\{document\}',
        
        # Bash code patterns
        r'#!/bin/bash', r'\bfi\b', r'\bdone\b', r'\bdo\b', r'\bthen\b', r'\belse\b', r'\bif\b', r'\bfor\b', r'\bwhile\b', 
        r'\becho\b', r'\bcd\b', r'\bexport\b', r'\bdeclare\b', r'\blet\b', r'\bfunction\b', r'\bexit\b', r'\btrap\b', 
        r'^\$', r'^[A-Za-z_]+=',

        # Git command patterns
        r'\bgit\b', r'\bclone\b', r'\bcommit\b', r'\bpush\b', r'\bpull\b', r'\bcheckout\b', r'\bbranch\b', r'\bmerge\b', 
        r'\brebase\b', r'\binit\b', r'\bstatus\b', r'\blog\b', r'\bremote\b', r'\badd\b', r'\bstash\b', r'\breset\b'
    ]
    
    # Skip lines containing the word 'null' (case-insensitive)
    if re.search(r'\bnull\b', line, re.IGNORECASE):
        return True
    
    # Skip lines matching any code pattern
    if any(re.search(pattern, line) for pattern in code_patterns):
        return True  # Skip the message if it matches any code pattern
    elif re.match(r'^[א-ת]', line):
        return False  # Do not skip if it starts with a Hebrew letter
    else:
        return False  # Do not skip other lines

def main(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"Input file '{input_file}' not found!")
        sys.exit(1)

    output = []
    skip_message = False

    for line in lines:
        # Check if the line is the start of a new message
        if re.match(r'^[0-9]{1,2}/[0-9]{1,2}/[0-9]{2},', line):
            # If we're currently in a message to be skipped, reset the flag
            if skip_message:
                skip_message = False
        
        # Determine if the current line should cause the message to be skipped
        if should_skip_line(line):
            skip_message = True
        
        # Append the line to the output if we're not skipping the message
        if not skip_message:
            output.append(line)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(output)

    print(f"Messages containing code removed from {input_file} and saved to {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    main(input_file, output_file)
