import os
import json
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----- Configuration -----
FOLDER_PATH = "screenshots"  # Folder containing your screenshots
USER_LABELS = ["meeting", "shopping", "travel", "meme", "note"]  # Customize these
OCR_LANG = "eng"  # Set to "eng" or install/use others via Tesseract if needed
OUTPUT_RAW_JSON = "raw_texts.json"
OUTPUT_CLUSTERED_JSON = "clustered_texts.json"

# Set Tesseract executable path for macOS
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# ----- Step 1: Extract text from images -----
def extract_text_from_images(folder_path):
    data = []
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return data
        
    image_count = 0
    success_count = 0
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            image_count += 1
            try:
                print(f"Processing {filename}...")
                img = Image.open(img_path)
                text = pytesseract.image_to_string(img, lang=OCR_LANG)
                if text.strip():
                    success_count += 1
                data.append({
                    "filename": filename,
                    "text": text.strip()
                })
                print(f"  Text extracted: {text[:50]}..." if len(text) > 50 else f"  Text extracted: {text}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"Processed {image_count} images, successfully extracted text from {success_count}")
    return data

# ----- Step 2: Save extracted text -----
def save_as_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

# ----- Step 3: Cluster by similarity to user-defined labels -----
def cluster_by_labels(text_data, user_labels):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    label_embeddings = model.encode(user_labels)

    clustered_data = {label: [] for label in user_labels}

    for item in text_data:
        text = item['text']
        if not text.strip():
            continue  # skip blank OCRs

        text_embedding = model.encode(text)
        similarities = cosine_similarity([text_embedding], label_embeddings)[0]
        best_match_idx = int(np.argmax(similarities))
        best_label = user_labels[best_match_idx]

        clustered_data[best_label].append(item)

    return clustered_data

# ----- Step 4: Run full pipeline -----
def main():
    print("🔍 Extracting text from screenshots...")
    extracted_data = extract_text_from_images(FOLDER_PATH)
    save_as_json(extracted_data, OUTPUT_RAW_JSON)
    print(f"✅ Raw OCR data saved to {OUTPUT_RAW_JSON}")

    print("🧠 Clustering using SentenceTransformer...")
    clustered = cluster_by_labels(extracted_data, USER_LABELS)
    save_as_json(clustered, OUTPUT_CLUSTERED_JSON)
    print(f"✅ Clustered data saved to {OUTPUT_CLUSTERED_JSON}")

if __name__ == "__main__":
    main()
