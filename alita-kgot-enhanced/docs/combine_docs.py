#!/usr/bin/env python3
"""
Documentation Combiner Script
Combines all documentation files in the docs directory into a single comprehensive document.
"""

import os
import glob
from pathlib import Path
from datetime import datetime

def read_file_content(file_path):
    """Read content from a file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"

def combine_docs():
    """Combine all documentation files into one."""
    docs_dir = Path(__file__).parent
    output_file = docs_dir / "COMBINED_DOCUMENTATION.md"
    
    # Get all markdown files in the docs directory
    md_files = []
    
    # Main directory files
    for file in sorted(docs_dir.glob("*.md")):
        if file.name != "COMBINED_DOCUMENTATION.md":  # Don't include the output file
            md_files.append(file)
    
    # API directory files
    api_dir = docs_dir / "api"
    if api_dir.exists():
        for file in sorted(api_dir.glob("*.md")):
            md_files.append(file)
    
    # Architecture directory files
    arch_dir = docs_dir / "architecture"
    if arch_dir.exists():
        for file in sorted(arch_dir.glob("*.md")):
            md_files.append(file)
    
    # Deployment directory files
    deploy_dir = docs_dir / "deployment"
    if deploy_dir.exists():
        for file in sorted(deploy_dir.glob("*.md")):
            md_files.append(file)
    
    # Start building the combined document
    combined_content = []
    
    # Add header
    combined_content.append("# Alita-KGoT Enhanced System - Complete Documentation")
    combined_content.append("")
    combined_content.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    combined_content.append("")
    combined_content.append("This document combines all documentation from the Alita-KGoT Enhanced System project.")
    combined_content.append("")
    
    # Add table of contents
    combined_content.append("## Table of Contents")
    combined_content.append("")
    
    toc_entries = []
    for i, file_path in enumerate(md_files, 1):
        relative_path = file_path.relative_to(docs_dir)
        section_name = file_path.stem.replace('_', ' ').replace('-', ' ').title()
        toc_entries.append(f"{i}. [{section_name}](#{section_name.lower().replace(' ', '-')})")
    
    combined_content.extend(toc_entries)
    combined_content.append("")
    combined_content.append("---")
    combined_content.append("")
    
    # Add each document
    for i, file_path in enumerate(md_files, 1):
        relative_path = file_path.relative_to(docs_dir)
        section_name = file_path.stem.replace('_', ' ').replace('-', ' ').title()
        
        print(f"Processing: {relative_path}")
        
        # Add section header
        combined_content.append(f"## {i}. {section_name}")
        combined_content.append("")
        combined_content.append(f"**Source:** `{relative_path}`")
        combined_content.append("")
        
        # Read and add file content
        content = read_file_content(file_path)
        
        # Remove the first # header from the content if it exists to avoid duplicate headers
        lines = content.split('\n')
        if lines and lines[0].startswith('# '):
            lines = lines[1:]
            # Remove empty lines after the removed header
            while lines and lines[0].strip() == '':
                lines.pop(0)
        
        combined_content.append('\n'.join(lines))
        combined_content.append("")
        combined_content.append("---")
        combined_content.append("")
    
    # Write the combined document
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(combined_content))
        
        print(f"\nSuccessfully combined {len(md_files)} documentation files into: {output_file}")
        combined_text = '\n'.join(combined_content)
        print(f"Total size: {len(combined_text)} characters")
        
    except Exception as e:
        print(f"Error writing combined documentation: {str(e)}")

if __name__ == "__main__":
    combine_docs()