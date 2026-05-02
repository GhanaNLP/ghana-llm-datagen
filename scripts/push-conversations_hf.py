import pandas as pd
from datasets import Dataset, Features, Value
from huggingface_hub import login

# ── CONFIG ─────────────────────────────────────────────
CSV_FILE = "combined_output.csv"
HF_TOKEN = "HF_TOKEN_HERE"  # <-- Use your active HF token
REPO_ID = "ghananlpcommunity/ghana-chat"
PRIVATE = False
# ──────────────────────────────────────────────────────

# 1. Load and sanitize CSV
print(f"Loading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE, low_memory=False)
df = df.fillna("")
df = df.astype(str)

# Ensure turn_number is treated as an integer for correct chronological sorting
df['turn_number'] = pd.to_numeric(df['turn_number'], errors='coerce').fillna(0).astype(int)

# 2. Sort the data so conversations are in the correct chronological order
print("Sorting data by conversation chunks and turns...")
df = df.sort_values(by=['id', 'chunk_id', 'turn_number'])

# 3. Create the individual message dictionaries
print("Building message structures...")
df['message_dict'] = df.apply(lambda row: {"role": row['role'], "content": row['content']}, axis=1)

# 4. Group by BOTH 'id' and 'chunk_id' to form real conversations
# Notice we only keep 'source_title' now. We are dropping category, url, and date.
print("Grouping rows into multi-turn conversations (this may take a minute)...")
grouped = df.groupby(['id', 'chunk_id']).agg({
    'source_title': 'first',
    'message_dict': list  # Groups the individual turns into a list
}).reset_index()

# 5. Clean up columns and data
grouped = grouped.rename(columns={'message_dict': 'messages'})

# Create the unique ID
grouped['unique_conv_id'] = grouped['id'] + "_" + grouped['chunk_id']

# Clean the source_title by removing " - MyJoyOnline"
print("Cleaning source titles...")
grouped['source_title'] = grouped['source_title'].str.replace(' - MyJoyOnline', '', regex=False).str.strip()

# Keep ONLY the columns we want to push
grouped = grouped[['unique_conv_id', 'source_title', 'messages']]

# 6. Define strict Hugging Face features
features = Features({
    "unique_conv_id": Value("string"),
    "source_title": Value("string"),
    "messages": [
        {
            "role": Value("string"),
            "content": Value("string")
        }
    ]
})

# 7. Convert to Hugging Face Dataset and Push
dataset = Dataset.from_pandas(grouped, features=features)
print(f"Successfully reconstructed {len(dataset)} multi-turn conversations!")

print("Pushing to Hugging Face...")
login(token=HF_TOKEN)

dataset.push_to_hub(
    REPO_ID,
    private=PRIVATE,
    token=HF_TOKEN
)

print(f"\n✓ Pushed cleaned multi-turn dataset to https://huggingface.co/datasets/{REPO_ID}")
