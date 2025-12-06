from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import json

PAD_IDX = 0  # padding value

# Load dataset
ds = load_dataset("Rizqi/emotion-raw")
df = pd.DataFrame(ds['train'])

# Save raw dataset copy
df.to_csv("emotion_raw.csv", index=False)

emotions_mapped = {}
emotions = []
sentences = []

# First pass: build list of sentences + emotions
for _, row in tqdm(df.iterrows(), total=len(df)):

    emotion = row["Emotion"]     # column name fix
    text = row["Text"]           # column name fix

    # Build emotion → id mapping
    if emotion not in emotions_mapped:
        emotions_mapped[emotion] = len(emotions_mapped)

    emotions.append(emotions_mapped[emotion])

    # Convert characters → ord() list
    sentence = [ord(c) for c in text]
    sentences.append(sentence)

# ----------------------------------------------
# 🔥 AUTO-PADDING
# ----------------------------------------------

max_len = max(len(s) for s in sentences)

def pad_sequence(seq, max_len, pad_value=PAD_IDX):
    if len(seq) < max_len:
        return seq + [pad_value] * (max_len - len(seq))
    else:
        return seq[:max_len]

# Apply padding
padded_sentences = [pad_sequence(s, max_len) for s in sentences]

# ----------------------------------------------
# SAVE PROCESSED DATA
# ----------------------------------------------

df = pd.DataFrame({
    "Emotion": emotions,
    "Text": padded_sentences
})
df.to_csv("processed_emotions.csv", index=False)

with open("mappings.json", "w") as wi:
    json.dump(emotions_mapped, wi, indent=4)

print("✔ Processing done!")
print(f"Max sentence length: {max_len}")
print(f"Total classes: {len(emotions_mapped)}")
