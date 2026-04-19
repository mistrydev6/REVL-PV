import re
import torch
from collections import Counter
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_PATH = "YOUR MODEL HERE"

TEST_IMAGE_PATH = "YOUR IMAGE PATH HERE"
TEST_INSTRUCTION = "Analyze this solar panel image for defects."

ALLOWED_LABELS = [
    "CRACK", "CLEAN PANEL", "FINGER", "BLACK CORE", "THICK LINE",
    "HORIZONTAL DISLOCATION", "VERTICAL DISLOCATION", "SHORT CIRCUIT",
]

NORMALIZE = {
    "CLEAN": "CLEAN PANEL", "CLEAN-PANEL": "CLEAN PANEL", "CLEAN_PANEL": "CLEAN PANEL", 
    "CLEAN PANEL.": "CLEAN PANEL", "CLEANPANEL": "CLEAN PANEL",
    "FINGER DEFECT": "FINGER", "FINGERS": "FINGER", "CRACKS": "CRACK", "STAR CRACK": "CRACK",
    "BLACK_CORE": "BLACK CORE", "BLACK SPOT": "BLACK CORE", "BLACK POINT": "BLACK CORE",
    "SHORT-CIRCUIT": "SHORT CIRCUIT", "SHORTCIRCUIT": "SHORT CIRCUIT",
    "THICK_LINE": "THICK LINE", "HORIZONTAL DIISLOCATION": "HORIZONTAL DISLOCATION", 
    "VERTICAL DIISLOCATION": "VERTICAL DISLOCATION",
    "MICROCRACKS": "CRACK", "SHADINGS": "CLEAN PANEL", 
    "HOT SPOT": "BLACK CORE", "SOILING": "CLEAN PANEL",
    "SHUNT DEFECT": "SHORT CIRCUIT"
}

SYSTEM_PROMPT = '''You are a world-class solar panel defect analyst. When analyzing provided image <image>, first think through your detailed analysis process internally, then provide a detailed and concise answer with specific probability metrics. Always answer in English only.

**CRITICAL INSTRUCTION: You must THINK in English and ANSWER in English.**

<think>
[Analysis Phase]
In this section, conduct your detailed analysis:
- Step 1: Carefully examine the image for damage.
- Step 2: Consider common defect patterns.
- Step 3: Assess potential causes.
- Step 4: Estimate the most likely defect category.
- Step 5: Assign probability percentages.
</think>

<answer>
[Final Diagnosis]
- **Defect Type**: [Name of the most likely defect]
- **Defect Category Probabilities**:
  - [Defect A]: XX%
  - [Defect B]: XX%
  - [Defect C]: XX%
- **Occurrence Probability in Solar Installations**: XX%
- **Most Likely Cause**: [...]
- **Supporting Evidence**:
  - Visual traits observed: [...]
- **Recommendation**: [...]
</answer>'''

def canon_label(s: str) -> str:
    if not s: return ""
    t = s.strip().upper()
    t = re.sub(r"[\-_/]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = NORMALIZE.get(t, t)
    for lbl in sorted(ALLOWED_LABELS, key=len, reverse=True):
        if t.startswith(lbl): return lbl
    return t

def extract_label_from_answer(text: str) -> str:
    if not text: return ""
    for line in text.splitlines():
        if "- **Defect Type**" in line or "- **defect type**" in line.lower():
            parts = line.split(":", 1)
            return canon_label(parts[1] if len(parts) > 1 else "")
    text_upper = text.upper()
    found = [lbl for lbl in ALLOWED_LABELS if lbl in text_upper]
    return max(found, key=len) if found else ""

def make_six_views(im: Image.Image):
    target = 672
    w, h = im.size
    # Full Image
    v_full = im.resize((target, target)) 
    # Crops
    crop_sz = min(w, h)
    c_x, c_y = (w - crop_sz) // 2, (h - crop_sz) // 2
    v_center = im.crop((c_x, c_y, c_x + crop_sz, c_y + crop_sz))
    v_tl     = im.crop((0, 0, crop_sz, crop_sz))
    v_tr     = im.crop((w - crop_sz, 0, w, crop_sz))
    v_bl     = im.crop((0, h - crop_sz, crop_sz, h))
    v_br     = im.crop((w - crop_sz, h - crop_sz, w, h))
    return [v_full, v_center, v_tl, v_tr, v_bl, v_br]

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
print("Model loaded successfully.")

@torch.inference_mode()
def run_tta_inference(image_path: str, instruction: str):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"Error loading image: {e}", ""

    views = make_six_views(img)

    texts = []
    image_inputs = []
    
    for view in views:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": view},
                {"type": "text", "text": instruction}
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        image_inputs.append(view)

    inputs = processor(
        text=texts,
        images=image_inputs, 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=768, 
        do_sample=False
    )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_preds = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    
    view_results = []
    for i, pred_text in enumerate(batch_preds):
        lbl = extract_label_from_answer(pred_text)
        view_results.append({"view": i, "label": lbl, "text": pred_text})
        
    global_res = view_results[0]
    crop_res   = view_results[1:]
    

    valid_crop_lbls = [r['label'] for r in crop_res if r['label'] in ALLOWED_LABELS]
    if valid_crop_lbls:
        crop_majority = Counter(valid_crop_lbls).most_common(1)[0][0]
    else:
        crop_majority = "CLEAN PANEL"

    final_text = ""
    
    if global_res['label'] == crop_majority:
        final_text = global_res['text']
        final_label = global_res['label']
    else:
        is_global_defect = global_res['label'] != "CLEAN PANEL"
        is_crop_defect = crop_majority != "CLEAN PANEL"

        if is_crop_defect and not is_global_defect:
            winner = next(r for r in crop_res if r['label'] == crop_majority)
            final_text = winner['text']
            final_label = winner['label']

        elif is_global_defect and not is_crop_defect:
            any_crop_confirms = any(
                r['label'] == global_res['label'] for r in crop_res
            )
            if any_crop_confirms:
                final_text = global_res['text']
                final_label = global_res['label']
            else:
                clean_crop = next(
                    (r for r in crop_res if r['label'] == "CLEAN PANEL"), crop_res[0]
                )
                final_text = clean_crop['text']
                final_label = "CLEAN PANEL"

        else:
            winner = next((r for r in crop_res if r['label'] == crop_majority), crop_res[0])
            final_text = winner['text']
            final_label = winner['label']
            
    return final_label, final_text

if __name__ == "__main__":
    print(f"Running inference on: {TEST_IMAGE_PATH}")
    predicted_label, response_text = run_tta_inference(TEST_IMAGE_PATH, TEST_INSTRUCTION)
    
    print("\n" + "="*40)
    print(f"FINAL PREDICTED LABEL: {predicted_label}")
    print("="*40 + "\n")
    print(response_text)