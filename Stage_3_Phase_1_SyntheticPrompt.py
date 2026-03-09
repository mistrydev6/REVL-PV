
import json
import os
import random
import time 
import requests 


try:
    api_key = 'GEMINI API KEY'
except KeyError:
    print("Error: GEMINI_API_KEY environment variable not set. Set it and rerun.")
    exit(1) 
api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}" #C hange to your desired model

defect_classes = [
    "CRACK", "CLEAN PANEL", "FINGER", "BLACK CORE",
    "THICK LINE", "HORIZONTAL DISLOCATION", "VERTICAL DISLOCATION", "SHORT CIRCUIT"
]

# Choose how many data points you want to generate per class
defect_counts = {
    "CRACK": 1,
    "CLEAN PANEL": 1,
    "FINGER": 1,
    "BLACK CORE": 1,
    "THICK LINE": 1,
    "HORIZONTAL DISLOCATION": 1,
    "VERTICAL DISLOCATION": 1,
    "SHORT CIRCUIT": 1
}

system_prompt_base = """
You are a world-class solar panel defect analyst. When analyzing a described defect scenario, first think through your detailed analysis process internally, then provide a detailed and concise answer with specific probability metrics. Always answer in English only. Always follow the structure below, don't deviate from it.

<think>
[Analysis Phase]
In this section, conduct your detailed analysis, giving answers to each step as part of your thinking process:
- Step 1: Carefully examine the described features for any indicators of damage or irregularities.
- Step 2: Consider common defect patterns. Just focus on the defect (e.g., cracks, dislocations, hotspots, etc.).
- Step 3: Assess potential causes (e.g., manufacturing error, mechanical stress, weather conditions).
- Step 4: Estimate the most likely defect category.
- Step 5: Assign **probability percentages** to the top 3 most likely defect types.
- Step 6: Estimate the **occurrence probability** of the predicted defect in real-world installations.
- Step 7: Reference any known patterns or characteristics from domain expertise.
</think>

<answer>
[Final Diagnosis]
In this section, provide your final assessment, using the knowledge gained from the thinking process to fill the rows here:
- **Defect Type**: [Name of the most likely defect]
- **Defect Category Probabilities**:
  - [Defect A]: XX%
  - [Defect B]: XX%
  - [Defect C]: XX%
- **Occurrence Probability in Solar Installations**: XX%
- **Most Likely Cause**: [Describe the cause and confidence level]
- **Supporting Evidence**:
  - Described traits: [...]
  - Panel region affected: [...]
  - Approximate area affected: XX cm²
- **Recommendation**:
  - [Optional remediation or next steps]
- **Background Info**:
  - [Relevant domain context, manufacturing insights, or frequency data]
</answer>
"""

scenario_templates = {
    "CRACK": [
        "Linear fractures across multiple silicon cells, likely from sudden mechanical impact during installation.",
        "Radial spiderweb patterns originating from a central point, caused by hail damage in stormy weather.",
        "Micro-cracks barely visible, resulting from thermal cycling in extreme temperature fluctuations.",
        "Severe branching cracks along grain boundaries, induced by manufacturing stress.",
        "Isolated crack in one cell, combined with minor discoloration from weathering.",
        "Propagating cracks from edge to center, due to improper handling during transport.",
        "Hidden micro-cracks under normal light, revealed by electroluminescence imaging.",
        "Cluster of parallel cracks, stemming from vibration in high-wind environments.",
        "Diagonal cracks intersecting busbars, caused by module flexing under snow load.",
        "Fine hairline cracks with slight chipping, from material defects in production.",
        "Extensive cracking across the panel, exacerbated by long-term UV exposure.",
        "Combined crack and delamination at edges, from moisture ingress in humid climates.",
        "Star-shaped crack from point impact, with potential for hotspot formation.",
        "Intermittent cracks in polycrystalline cells, linked to lamination errors.",
        "Widespread micro-cracks, simulating aging after 10 years of operation."
    ],
    "CLEAN PANEL": [
        "Uniform surface with intact grid lines and no visible irregularities, post-cleaning.",
        "Pristine monocrystalline cells showing even color distribution, in ideal lab conditions.",
        "Slightly dusty but otherwise flawless panel, after routine maintenance in a dry climate.",
        "Clear panel with minor surface scratches that don't affect performance.",
        "Intact polycrystalline structure, free from cracks or spots after factory inspection.",
        "Panel with even reflectivity, no signs of degradation after short-term exposure.",
        "Clean edges and busbars, simulating a newly installed module.",
        "Uniform blue hue across cells, with no hotspots or dislocations evident.",
        "Panel in shaded conditions, maintaining clarity without thermal stress marks.",
        "High-efficiency panel with no manufacturing flaws, under standard testing.",
        "Clean but aged panel, showing only faint wear from environmental factors.",
        "Pristine surface combined with minor pollen accumulation, easily removable.",
        "Uniform grid without interruptions, post-washing in rainy weather.",
        "Clear cells with no internal defects, as seen in quality control checks.",
        "Panel with perfect alignment, free from any electrical anomalies."
    ],
    "FINGER": [
        "Broken finger lines in the upper cells, from screen-printing errors in manufacturing.",
        "Discontinuous finger contacts across the panel, caused by corrosion over time.",
        "Missing fingers at cell edges, due to mechanical abrasion during assembly.",
        "Irregular finger thickness, leading to uneven current collection.",
        "Fractured fingers combined with minor cracks, from thermal expansion mismatches.",
        "Faded finger metallization, resulting from chemical exposure in polluted areas.",
        "Partial finger interruptions in central areas, simulating production defects.",
        "Overlapped or misaligned fingers, caused by alignment issues in printing.",
        "Thin, breaking fingers under stress, from high-current operation.",
        "Finger defects clustered in one quadrant, linked to lamination pressure.",
        "Eroded fingers with residue buildup, from improper cleaning agents.",
        "Combined finger and short circuit risks, in humid environments.",
        "Uneven finger spacing, affecting efficiency in large-scale arrays.",
        "Brittle fingers cracking under vibration, as in rooftop installations.",
        "Subtle finger discontinuities, only visible under magnification."
    ],
    "BLACK CORE": [
        "Large dark spots in cell centers, indicating severe overheating from shading.",
        "Small black cores at edges, caused by material impurities in production.",
        "Irregular black areas spreading from hotspots, due to mismatched cells.",
        "Black core combined with cracking, from prolonged thermal stress.",
        "Faint black discoloration in multiple cells, simulating early degradation.",
        "Centralized black core with halo effect, linked to electrical imbalances.",
        "Black spots from chemical reactions, in contaminated manufacturing.",
        "Expanding black core under UV light, exacerbating power loss.",
        "Isolated black core in one cell, from impact damage.",
        "Black core patterns in polycrystalline panels, due to grain boundary issues.",
        "Combined black core and delamination, in high-humidity settings.",
        "Dark core with surrounding fading, caused by reverse current flow.",
        "Subtle black spots emerging over time, from operational wear.",
        "Black core clusters, indicating systemic manufacturing flaws.",
        "Localized blackening with heat signatures, from partial shading."
    ],
    "THICK LINE": [
        "Wide dark lines horizontally across busbars, from soldering inconsistencies.",
        "Thick vertical lines disrupting current flow, caused by printing errors.",
        "Irregular thick lines with branching, due to mechanical pressure.",
        "Thick line combined with finger breaks, in high-temperature environments.",
        "Faint thick lines along edges, simulating lamination defects.",
        "Overly thick busbar lines, leading to shadowing effects.",
        "Thick lines intersecting cells, from material buildup in production.",
        "Wavy thick lines, caused by vibration during transport.",
        "Thick line defects in central areas, exacerbating resistance.",
        "Combined thick line and crack patterns, under snow load.",
        "Subtle thick lines visible only in EL imaging, from thermal expansion.",
        "Thick horizontal lines with discoloration, in polluted atmospheres.",
        "Vertical thick lines clustering, indicating alignment issues.",
        "Thick lines with chipping, from abrasive cleaning.",
        "Extensive thick line networks, simulating aged panels."
    ],
    "HORIZONTAL DISLOCATION": [
        "Step-like horizontal shifts in grid lines, from lamination pressure.",
        "Misaligned horizontal busbars, caused by manufacturing tolerances.",
        "Horizontal dislocation combined with cracks, due to thermal cycling.",
        "Subtle horizontal offsets in cells, simulating installation stress.",
        "Pronounced horizontal dislocations across the panel, from flexing.",
        "Horizontal shifts at edges, linked to moisture expansion.",
        "Clustered horizontal dislocations, indicating material defects.",
        "Horizontal misalignment with resistance increase, in windy conditions.",
        "Faint horizontal dislocations, visible under stress testing.",
        "Combined horizontal dislocation and black spots, from overheating.",
        "Horizontal shifts propagating over time, in aging modules.",
        "Irregular horizontal dislocations, caused by vibration.",
        "Horizontal offsets in polycrystalline cells, from grain issues.",
        "Severe horizontal dislocation with power drop, post-impact.",
        "Horizontal dislocations in shaded areas, exacerbating hotspots."
    ],
    "VERTICAL DISLOCATION": [
        "Vertical shifts in cell alignment, from vertical stress during assembly.",
        "Pronounced vertical dislocations along busbars, caused by expansion.",
        "Vertical dislocation combined with thick lines, due to mechanical load.",
        "Subtle vertical offsets, simulating thermal mismatches.",
        "Vertical misalignments clustering in centers, from production flaws.",
        "Vertical shifts at panel edges, linked to humidity changes.",
        "Irregular vertical dislocations, exacerbating current imbalances.",
        "Vertical offsets with cracking risks, in high-vibration setups.",
        "Faint vertical dislocations, only apparent in EL views.",
        "Combined vertical dislocation and finger defects, from corrosion.",
        "Propagating vertical shifts over years, in outdoor installations.",
        "Vertical misalignments causing hotspots, under partial shading.",
        "Severe vertical dislocations with delamination, post-storm.",
        "Vertical offsets in monocrystalline panels, from crystal defects.",
        "Extensive vertical dislocation networks, indicating systemic issues."
    ],
    "SHORT CIRCUIT": [
        "Abnormal conduction paths in central cells, from solder bridges.",
        "Short circuit spots with heating, caused by insulation failure.",
        "Short circuits combined with black cores, due to moisture ingress.",
        "Localized short circuits at edges, simulating wiring errors.",
        "Intermittent short circuits, from loose connections in arrays.",
        "Short circuit clusters, linked to manufacturing contaminants.",
        "Severe short circuits leading to arcing, in high-voltage setups.",
        "Faint short circuit indications, visible in thermal imaging.",
        "Short circuits with resistance spikes, from thermal stress.",
        "Combined short circuit and cracks, exacerbating power loss.",
        "Short circuits propagating along busbars, in humid climates.",
        "Isolated short circuits in one cell, from impact damage.",
        "Short circuit risks in shaded areas, causing bypass issues.",
        "Widespread short circuits, indicating batch defects.",
        "Subtle short circuits emerging over time, from degradation."
    ]
}

def get_scenario(defect_class):
    templates = scenario_templates.get(defect_class, ["Generic defect description."])
    return random.choice(templates)

def generate_synthetic_dataset(output_file='fre_synthetic_data.jsonl'):
    synthetic_data = []
    total_samples = sum(defect_counts.values())
    progress_counter = 0

    for defect_class in defect_classes:
        num_samples = defect_counts.get(defect_class, 0)
        for i in range(num_samples):
            scenario = get_scenario(defect_class)
            user_prompt = f"{system_prompt_base}\n\nAnalyze this described solar panel defect: {scenario}"

            try:
                request_body = {
                    "contents": [
                        {
                            "parts": [
                                {"text": user_prompt}
                            ]
                        }
                    ]
                }

                response = requests.post(api_url, json=request_body, headers={"Content-Type": "application/json"})

                if response.status_code == 200:
                    response_data = response.json()
                    generated_answer = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
                else:
                    raise Exception(f"API request failed with status {response.status_code}: {response.text}")

            except Exception as e:
                print(f"API error for {defect_class} (sample {i+1}/{num_samples}): {e}. Using placeholder.")
                generated_answer = f"<think>Placeholder analysis for {defect_class}.</think><answer>[Final Diagnosis] - **Defect Type**: {defect_class}</answer>"

            message = json.dumps([{"role": "user", "content": [{"type": "text", "text": user_prompt}]}])
            entry = {"message": message, "answer": generated_answer}
            synthetic_data.append(entry)

            progress_counter += 1
            if progress_counter % 100 == 0:
                print(f"Progress: Generated {progress_counter}/{total_samples} samples.")
            time.sleep(1) 


    random.shuffle(synthetic_data)


    with open(output_file, 'w') as f:
        for entry in synthetic_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Generated {len(synthetic_data)} samples saved to {output_file}.")

generate_synthetic_dataset()

