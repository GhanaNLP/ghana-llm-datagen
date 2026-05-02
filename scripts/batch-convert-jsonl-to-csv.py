import json
import csv
from pathlib import Path

input_folder = Path('jsonl_files')  # <-- change to your folder path
output_file = 'combined_output.csv'

# Collect all JSONL files
jsonl_files = sorted(input_folder.glob('*.jsonl'))

if not jsonl_files:
    print(f'No .jsonl files found in {input_folder}')
    exit(1)

with open(output_file, 'w', newline='', encoding='utf-8') as out:
    writer = csv.writer(out)
    writer.writerow(['id', 'source_title', 'category', 'source_url', 'source_date', 'chunk_id', 'turn_number', 'role', 'content'])
    
    total_rows = 0
    for input_file in jsonl_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                base = [data.get(k, '') for k in ['id', 'source_title', 'category', 'source_url', 'source_date', 'chunk_id']]
                
                for i, msg in enumerate(data.get('conversations', []), 1):
                    writer.writerow(base + [i, msg.get('role', ''), msg.get('content', '')])
                    total_rows += 1
        
        print(f'  ✓ Processed: {input_file.name}')

print(f'\n✓ Combined {len(jsonl_files)} files into: {output_file}')
print(f'  Total rows written: {total_rows}')
