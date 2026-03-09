import os
import base64
import json
import requests
from PIL import Image
from io import BytesIO
import mimetypes
import time

def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith('image'):
                mime_type = 'image/jpeg'

            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
                mime_type = 'image/jpeg'

            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_byte = buffered.getvalue()
            
            base64_string = base64.b64encode(img_byte).decode('utf-8')
            return base64_string, mime_type
            
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None, None

def analyze_image_with_gemini(image_path, api_key):
    print(f"Analyzing: {os.path.basename(image_path)}...")
    
    base64_image, mime_type = encode_image_to_base64(image_path)
    if not base64_image:
        return None

    prompt = ''' THESE ARE THE DEFECT CLASSES YOU CAN CHOOSE FOR THE IMAGE: [CRACK, CLEAN PANEL, FINGER, BLACK CORE, THICK LINE, HORIZONTAL DISLOCATION, VERTICAL DIISLOCATION, SHORT CIRCUIT]. START DIRECTLY WITH THE <think> block and do not mention this line.
    You are a world-class solar panel defect analyst. When analyzing an image, first think through your detailed analysis process internally, then provide a detailed and concise answer with specific probability metrics. Always answer in English only. Always follow the structure down below, don't deviate from the structure

<think>
[Analysis Phase]
In this section, conduct your detailed analysis, give answer to each and every step as part of your thinking process:
- Step 1: Carefully examine the image for any visible indicators of damage or irregularities.
- Step 2: Consider common defect patterns. Just focus on the defect (e.g., cracks, dislocations, hotspots, etc.).
- Step 3: Assess potential causes (e.g., manufacturing error, mechanical stress, weather conditions).
- Step 4: Estimate the most likely defect category.
- Step 5: Assign **probability percentages** to the top 3 most likely defect types.
- Step 6: Estimate the **occurrence probability** of the predicted defect in real-world installations.
- Step 7: Reference any known patterns or characteristics from domain expertise.
</think>

<answer>
[Final Diagnosis]
In tis section, provide your final assessment, use the knowledge gained from the thinking process to fill the rows here:
- **Defect Type**: [Name of the most likely defect]
- **Defect Category Probabilities**:
  - [Defect A]: XX%
  - [Defect B]: XX%
  - [Defect C]: XX%
- **Occurrence Probability in Solar Installations**: XX%
- **Most Likely Cause**: [Describe the cause and confidence level]
- **Supporting Evidence**:
  - Visual traits observed: [...]
  - Panel region affected: [...]
  - Approximate area affected: XX cm²
- **Recommendation**:
  - [Optional remediation or next steps]
- **Background Info**:
  - [Relevant domain context, manufacturing insights, or frequency data]
</answer>'''

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": mime_type, "data": base64_image}}
            ]
        }]
    }
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    max_retries = 5
    base_delay = 2 

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
            
            if response.status_code == 429:
                wait_time = base_delay * (2 ** attempt)
                print(f"Rate limit hit. Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                continue 
            
            response.raise_for_status() 
            
            result = response.json()
            if (result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts')):
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                print(f"Warning: Unexpected API response structure for {os.path.basename(image_path)}.")
                return "Analysis failed: The API response was not in the expected format."

        except requests.exceptions.RequestException as e:
            print(f"Error on attempt {attempt + 1} for {os.path.basename(image_path)}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None 
    
    print(f"Failed to process {os.path.basename(image_path)} after {max_retries} retries.")
    return None

def process_image_directory(directory_path, output_file, api_key):
    all_data = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(supported_extensions)]
    
    if not image_files:
        print(f"No images found in the directory: {directory_path}")
        return

    total_images = len(image_files)
    print(f"Found {total_images} images to process.")

    for i, filename in enumerate(image_files):
        image_path = os.path.join(directory_path, filename)
        
        analysis_output = analyze_image_with_gemini(image_path, api_key)

        if analysis_output:
            relative_image_path = os.path.join('train', 'JPEGImages', filename).replace('/', '\\')
            json_record = {
                "instruction": '''You are a world-class solar panel defect analyst. When analyzing an image, first think through your detailed analysis process internally, then provide a detailed and concise answer with specific probability metrics. Always answer in English only. Always follow the structure down below, don't deviate from the structure

<think>
[Analysis Phase]
In this section, conduct your detailed analysis, give answer to each and every step as part of your thinking process:
- Step 1: Carefully examine the image for any visible indicators of damage or irregularities.
- Step 2: Consider common defect patterns. Just focus on the defect (e.g., cracks, dislocations, hotspots, etc.).
- Step 3: Assess potential causes (e.g., manufacturing error, mechanical stress, weather conditions).
- Step 4: Estimate the most likely defect category.
- Step 5: Assign **probability percentages** to the top 3 most likely defect types.
- Step 6: Estimate the **occurrence probability** of the predicted defect in real-world installations.
- Step 7: Reference any known patterns or characteristics from domain expertise.
</think>

<answer>
[Final Diagnosis]
In tis section, provide your final assessment, use the knowledge gained from the thinking process to fill the rows here:
- **Defect Type**: [Name of the most likely defect]
- **Defect Category Probabilities**:
  - [Defect A]: XX%
  - [Defect B]: XX%
  - [Defect C]: XX%
- **Occurrence Probability in Solar Installations**: XX%
- **Most Likely Cause**: [Describe the cause and confidence level]
- **Supporting Evidence**:
  - Visual traits observed: [...]
  - Panel region affected: [...]
  - Approximate area affected: XX cm²
- **Recommendation**:
  - [Optional remediation or next steps]
- **Background Info**:
  - [Relevant domain context, manufacturing insights, or frequency data]
</answer>''',
                "input": "",
                "output": analysis_output,
                "images": [relative_image_path]
            }
            all_data.append(json_record)
        
        print(f"Progress: {i + 1}/{total_images} complete.")
        time.sleep(1.1)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
        print(f"\nProcessing complete. All data saved to {output_file}")
    except IOError as e:
        print(f"Error writing to output file {output_file}: {e}")

if __name__ == "__main__":
    GEMINI_API_KEY = "GEMINI API KEY"
    IMAGE_DIRECTORY = "DATASET DIRECTORY"
    OUTPUT_JSON_FILE = "OUTPUT DIRECTORY"

    if not os.path.isdir(IMAGE_DIRECTORY):
        print(f"Error: The specified image directory does not exist: {IMAGE_DIRECTORY}")
        print("Please update the 'IMAGE_DIRECTORY' variable in the script.")
    elif not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_NEW_API_KEY_HERE":
        print("Error: Please set your new Gemini API key in the GEMINI_API_KEY variable.")
    else:
        process_image_directory(IMAGE_DIRECTORY, OUTPUT_JSON_FILE, GEMINI_API_KEY)
