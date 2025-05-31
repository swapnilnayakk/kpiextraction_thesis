import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load dataset
def load_kpi_edgar(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)

def sentence_to_text(words: List[Dict[str, Any]]) -> str:
    return " ".join([w["value"] for w in words])

# Extract ground truth annotations
def extract_gold_kpi_values(sentence):
    entities = sentence["entities_anno"]
    words = sentence["words"]

    kpi_types = {"kpi"}
    value_types = {"cy", "py", "py1", "davon-cy", "davon-py"}

    entity_map = {i: e for i, e in enumerate(entities)}
    gold = []

    if not sentence.get("relations_anno"):
        return []  # Return an empty list if there are no relations
    for rel in sentence["relations_anno"]:
        h_idx, t_idx = rel["head_idx"], rel["tail_idx"]
        h, t = entity_map.get(h_idx), entity_map.get(t_idx)

        # Normal direction: KPI → value
        if h["type_"] in kpi_types and t["type_"] in value_types:
            k_span = (h["start"], h["end"])
            v_span = (t["start"], t["end"])
            v_type = t["type_"]

        # Reversed direction: value → KPI
        elif t["type_"] in kpi_types and h["type_"] in value_types:
            k_span = (t["start"], t["end"])
            v_span = (h["start"], h["end"])
            v_type = h["type_"]
        else:
            continue  # Skip unrelated pairs

        kpi_text = " ".join(w["value"] for w in words[k_span[0]:k_span[1]])
        val_text = " ".join(w["value"] for w in words[v_span[0]:v_span[1]])
        gold.append((kpi_text.lower(), val_text.lower(), v_type))

    return gold

class Data_Model:
    def __init__(self, input: str, completion: str):
        self.input = input
        self.completion = completion
        self.fixed_prompt = f"""
            You are a seasoned financial analyst, expert in financial statements review and extraction of KPI and its relations. You are given a financial statement of a company from a 10-K document filled with SEC. Your task is to extract the key performance indicators (KPIs) and their values from the text.
            Follow the below instructions carefully:
            1. Identify kpi (key performance indicator) names, which are finacial terms representing the economic state of the company.
            2. Extract the value associated with each KPI. Remove any currency symbols (e.g., $, €, ₹) or units (e.g., millions, billions) in the value.
            3. Determine the type of value. value_type can be one of the following: "cy" - current report year, "py" - previous year, "py1" - previous to previous year.

            Ensure that:
            - The KPI names are concise and meaningful. eg: "revenue", "marketable securities", "right-of-use assets or rou assets" etc.  Also, avoid verbosity or unnecessary details. For eg:, "net tax benefit related to changes to uncertain tax positions" could be "net tax benefit, "accumulated other comprehensive loss attributable to controlling interest" could be "comprehensive loss".
            - The values are extracted accurately without any additional formatting.
            - The value_type is assigned correctly based on the context of the sentence.

            All statement contains some KPIs and their values.
            Return the output in the format of a Python list of tuples. Each tuple should contain three elements: (kpi, value, value_type).
            Output format example: [("revenue", "100", "cy"), ("total costs", "50", "cy")]
            Analyze this <financial_statement>"{input}"</financial_statement> and extract the KPIs and their values. 
            strictly follow the output format and do not return any additional text.
            """

    def to_json(self) -> str:
        return json.dumps({
            "prompt": self.fixed_prompt, 
            "completion": self.completion
        })

def data_extraction(data: List[Dict[str, Any]]):
    results = list()
    for doc in data:
        for segment in doc["segments"]:
            sentences = segment.get("sentences", [])
            if sentences is None:
                continue
            for sentence in sentences:
                text = sentence_to_text(sentence["words"])
                gold = extract_gold_kpi_values(sentence)
                if not gold:
                    continue
                Data_Model_obj = Data_Model(input=text, completion=repr(gold))
                results.append(Data_Model_obj.to_json())
    
    length = len(results)
    training_data = results[:int(length * 0.6)]
    validation_data = results[int(length * 0.6):int(length * 0.8)]
    test_data = results[int(length * 0.8):]
    with open("extracted_results.jsonl", "w") as outfile:
        for result in results:
            outfile.write(result + "\n")

    with open("training_data.jsonl", "w") as outfile:
        for result in training_data:
            outfile.write(result + "\n")

    with open("validation_data.jsonl", "w") as outfile:
        for result in validation_data:
            outfile.write(result + "\n")

    with open("test_data.jsonl", "w") as outfile:
        for result in test_data:
            outfile.write(result + "\n")

if __name__ == "__main__":
    load_dotenv()
    dataset = load_kpi_edgar("kpi-edgar\data\kpi_edgar.json")  
    results = data_extraction(dataset)

 #   Get-Content .\extracted_results.json | ConvertFrom-Json | ForEach-Object { $_ | ConvertTo-Json -Depth 1 -Compress } > .\gold_annotations.jsonl