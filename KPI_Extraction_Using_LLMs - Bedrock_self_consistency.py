import json
from langchain.schema import HumanMessage
from typing import List, Tuple, Dict, Any
import re
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time

from langchain_aws import ChatBedrock

llm = ChatBedrock(
    credentials_profile_name="sf-test", 
    model_id="us.meta.llama4-scout-17b-instruct-v1:0",
    temperature=0.1,
    max_tokens=150,
    top_p=0.5,
    top_k=20,
)

# Load and parse dataset
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

def apply_self_consistency(text, responses: list[str]):
    results = ""
    for idx, response in enumerate(responses):
        results = results + f"""{idx}. {response} \n"""

    prompt = f"""
                You're a seasoned finacial analyst. You were tasked with extraction of KPI and it's relation from this finacial statement **{text}**.
                In python list of tuple format [(kpi, value, type)], where kpi are the financial terms, representing economic state of the company, eg: "revenue", "marketable securities", "right-of-use assets or rou assets" etc., do not include any articles or prepositions in kpi name, also, avoid verbosity or unnecessary details. 
                value is a string representing the value of the KPI. 
                value_type is a string representing the type of value. value_type can be one of the following: \"cy\" - current year, \"py\" previous year, \"py1\" previous to previous year\n\n
                Here are {len(responses)} provided by you earlier.
                {results}

                Example:
                Financial Statement: "The sale of this business resulted in a pre-tax gain of approximately $ 478 million , included in restructuring and other ( income ) costs , net."   
                Provided Results: 
                1. [("pre-tax gain", "478", "cy"), ("restructuring and other ( income ) costs", "net", "cy")]
                Output: [("pre-tax gain", "478", "cy")]

                Evaluate these results, select and return the best result from the earlier provided results.
                Do not merge the results or provide repetitive tuples, return only best result in python list of tuple format [(kpi, value, type)] only.
                Output format: [("revenue", "100", "cy"), ("right-of-use assets", "50", "py")]
                strictly follow the output format and do not return any additional text.
            """
    response = llm.invoke(prompt)
    return parse_llm_response(response.content)

# LangChain prompt interface
def query_llm(text: str) -> List[Tuple[str, str, str]]:
    prompt = f"""
            You are a seasoned financial analyst, expert in financial statements review and extraction of KPI and its relations. You are given a financial statement of a company from a 10-K document filled with SEC. Your task is to extract the key performance indicators (KPIs) and their values from the text.
            Follow the below instructions carefully:
            1. Identify kpi (key performance indicator) names, which are finacial terms representing the economic state of the company.
            2. Extract the value associated with each KPI. Remove any currency symbols (e.g., $, €, ₹) or units (e.g., millions, billions) in the value.
            3. Determine the type of value. value_type can be one of the following: "cy" - current report year, "py" - previous year, "py1" - previous to previous year.

            Ensure that:
            - The KPI names are concise and meaningful. eg: "revenue", "marketable securities", "right-of-use assets or rou assets" etc.  Also, avoid verbosity or unnecessary details. For eg:, "net tax benefit related to changes to uncertain tax positions" could be "net tax benefit, "accumulated other comprehensive loss attributable to controlling interest" could be "comprehensive loss".
            - The values are extracted accurately without any additional formatting.
            - The value_type is assigned correctly based on the context of the sentence.

            Here are some examples of input statements and their expected outputs:

            Input: "The sale of this business resulted in a pre-tax gain of approximately $ 478 million , included in restructuring and other ( income ) costs , net."   
            Output: [("pre-tax gain", "478", "cy")]

            Input: "Revenues in 2019 , through the date of sale , and the full year 2018 of the business sold were approximately $ 115 million and $ 238 million , respectively , net of retained sales through the company 's healthcare market and research and safety market channel businesses ."
            Output: [("revenue", "115", "cy"), ("revenue", "238", "py")]

            Input: "The total intrinsic value of options exercised during the same periods was $ 320 million , $ 312 million and $ 199 million , respectively ."
            Output: [("total intrinsic value of options exercised", "320", "cy"), ("total intrinsic value of options exercised", "199", "py1"), ("total intrinsic value of options exercised", "312", "py")]

            Input: "During 2020 , the company recorded net restructuring and other costs ( income ) by segment as follows : The principal components of net restructuring and other costs ( income ) by segment are as follows : Life Sciences Solutions In 2020 , the Life Sciences Solutions segment recorded $ 26 million of net restructuring and other charges"
            Output: [("net restructuring and other costs", "26", "cy")]

            Input: "Amortization expense during the years ended December 31 , 2019 , 2018 and 2017 was $ 44 million , $ 66 million and $ 40 million , respectively ."
            Output: [("amortization expense", "44", "cy"), ("amortization expense", "40", "py1"), ("amortization expense", "66", "py")]

            All statement contains some KPIs and their values.
            Return the output in the format of a Python list of tuples. Each tuple should contain three elements: (kpi, value, value_type).
            Output format: [("revenue", "100", "cy"), ("total costs", "50", "cy")]
            Analyze this <financial_statement>"{text}"</financial_statement> and extract the KPIs and their values. 
            strictly follow the output format and do not return any additional text.
            """

    response_list = []
    for i in range(3):
        response = llm.invoke(prompt)
        parsed_response = parse_llm_response(response.content)
        response_list.append(parsed_response)
    
    final_response = apply_self_consistency(text, response_list)

    return final_response

# parse LLM response to match list of tuples structure of output
def parse_llm_response(response: str) -> List[Tuple[str, str, str]]:
    try:
        pattern = r"\([^)]+\)"
        tuples = re.findall(pattern, response)
        result = []
        for t in tuples:
            parts = [p.strip().strip("'\"") for p in t.strip("()").split(",")]
            if len(parts) == 3:
                result.append(tuple(map(str.lower, parts)))
        return set(result)
    except Exception as e:
        print(f"Parse error: {e}\nResponse: {response}")
        return []

# Evaluation
def evaluate_predictions(gold: List[Tuple[str, str, str]], pred: List[Tuple[str, str, str]]) -> Tuple[float, float, float]:
    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    return precision, recall, f1

from typing import List, Tuple

def evaluate_partial_predictions(
    gold: List[Tuple[str, str, str]], 
    pred: List[Tuple[str, str, str]]
) -> Tuple[float, float, float]:
    """
    Partial match evaluation for kpi names and fixed match on value and type.
    """
    tp = 0  
    fp = 0 
    fn = 0 

    gold_set = set(gold)
    pred_set = set(pred)

    for pred_tuple in pred_set:
        pred_kpi, pred_value, pred_type = pred_tuple
        match_found = False

        # Check for partial match in gold tuples
        for gold_tuple in gold_set:
            gold_kpi, gold_value, gold_type = gold_tuple

            # do partial match on KPI name and exact match on value and type
            if pred_value == gold_value and pred_type == gold_type and (pred_kpi in gold_kpi) or (gold_kpi in pred_kpi):
                tp += 1
                match_found = True
                break

        if not match_found:
            fp += 1 

    # Count false negatives (gold tuples not matched by any prediction)
    for gold_tuple in gold_set:
        gold_kpi, gold_value, gold_type = gold_tuple
        match_found = False

        for pred_tuple in pred_set:
            pred_kpi, pred_value, pred_type = pred_tuple

            # Partial match on KPI name and exact match on value and type
            if pred_value == gold_value and pred_type == gold_type and (pred_kpi in gold_kpi) or (gold_kpi in pred_kpi):
                match_found = True
                break

        if not match_found:
            fn += 1  # False negative if no match is found

    # precision, recall, and F1 score
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)

    return precision, recall, f1


def calculate_similarity_score(gold: List[Tuple[str, str, str]], pred: List[Tuple[str, str, str]]) -> float:
    # Combine all elements of gold and pred into strings
    gold_str = " ".join([" ".join(item) for item in gold])
    pred_str = " ".join([" ".join(item) for item in pred])
    
    # Convert strings to vectors using TfidfVectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([gold_str, pred_str])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return similarity

# Main pipeline
def run_langchain_kpi_eval(data: List[Dict[str, Any]], max_sentences: int = 50):
    results = []
    cosine_similarity_scores = []
    count = 0
    for doc in data:
        if count > 0 and count % 10 == 0:
            time.sleep(60)  
        for segment in doc["segments"]:
            sentences = segment.get("sentences", [])
            if sentences:
                for sentence in sentences:
                    text = sentence_to_text(sentence["words"])
                    gold = extract_gold_kpi_values(sentence)
                    if not gold:
                        continue
                    pred = query_llm(text)
                    precision, recall, f1 = evaluate_partial_predictions(gold, pred)
                    similarity_score = calculate_similarity_score(gold, pred)
                    cosine_similarity_scores.append(similarity_score)
                    results.append((doc['id_'], text, gold, pred, similarity_score, precision, recall, f1))
                    
                    count += 1
                    # if count >= max_sentences:
                    #     return results, cosine_similarity_scores
    return results, cosine_similarity_scores

def print_results(results, result_df):

    avg_p, avg_r, avg_f = map(lambda x: sum(x)/len(x), zip(*[r[-3:] for r in results]))
    print(f"Precision: {avg_p:.2f}, Recall: {avg_r:.2f}, F1: {avg_f:.2f}")

    # Calculate precision, recall, and F1 score based on Cosine Similarity
    threshold = 0.65
    true_positives = sum(1 for score in result_df['Cosine Similarity'] if score > threshold)
    false_positives = len(result_df) - true_positives
    false_negatives = 0  # Assuming no false negatives for simplicity

    precision = true_positives / (true_positives + false_positives + 1e-5)
    recall = true_positives / (true_positives + false_negatives + 1e-5)
    f1_score = 2 * precision * recall / (precision + recall + 1e-5)

    print(f"Precision (Cosine Similarity > {threshold}): {precision:.2f}")
    print(f"Recall (Cosine Similarity > {threshold}): {recall:.2f}")
    print(f"F1 Score (Cosine Similarity > {threshold}): {f1_score:.2f}")

    if cosine_similarity_scores:
        max_similarity = max(cosine_similarity_scores)
        min_similarity = min(cosine_similarity_scores)
        avg_similarity = sum(cosine_similarity_scores) / len(cosine_similarity_scores)
        print(f"Cosine Similarity Scores: {', '.join(map(str, cosine_similarity_scores))}")
        print(f"Max Cosine Similarity: {max_similarity:.2f}")
        print(f"Min Cosine Similarity: {min_similarity:.2f}")
        print(f"Avg Cosine Similarity: {avg_similarity:.2f}")
    else:
        print("No cosine similarity scores to report.")

if __name__ == "__main__":
    load_dotenv()
    dataset = load_kpi_edgar("kpi-edgar\data\kpi_edgar.json")  
    results, cosine_similarity_scores = run_langchain_kpi_eval(dataset, max_sentences=5)

    # Create a dataset from results
    data = []
    for doc_id, text, gold, pred, similarity_score, precision, recall, f1 in results:
        data.append({
            "Document ID": doc_id,
            "Text": text,
            "Gold": gold,
            "Predicted": pred,
            "Cosine Similarity": similarity_score,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    # Convert to DataFrame
    result_df = pd.DataFrame(data)

    # Save to CSV
    output_path = "kpi_extraction_results_self_consistency_deepseek_r1.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print_results(results, result_df)
