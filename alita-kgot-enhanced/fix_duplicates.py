#!/usr/bin/env python3
"""
Script to remove ALL duplicate lines from kgot_controller.js
"""

def remove_all_duplicates(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    i = 0
    
    while i < len(lines):
        current_line = lines[i]
        
        # Keep adding identical consecutive lines until we find a different one
        consecutive_count = 1
        while (i + consecutive_count < len(lines) and 
               lines[i + consecutive_count].strip() == current_line.strip() and 
               current_line.strip() != ''):
            consecutive_count += 1
        
        # Add only one copy of the line
        cleaned_lines.append(current_line)
        
        # Skip all the duplicates
        i += consecutive_count
    
    with open(output_file, 'w') as f:
        f.writelines(cleaned_lines)
    
    print(f"Cleaned {len(lines)} lines down to {len(cleaned_lines)} lines")
    print(f"Removed {len(lines) - len(cleaned_lines)} duplicate lines")

if __name__ == "__main__":
    input_file = "/Users/radhikakulkarni/Downloads/kgot_alita/alita-kgot-enhanced/kgot_core/controller/kgot_controller.js.backup"
    output_file = "/Users/radhikakulkarni/Downloads/kgot_alita/alita-kgot-enhanced/kgot_core/controller/kgot_controller.js"
    
    remove_all_duplicates(input_file, output_file)
    print("File cleaned successfully!")