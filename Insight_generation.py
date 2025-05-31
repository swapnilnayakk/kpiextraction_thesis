import json
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
import pandas as pd
from langchain_aws import ChatBedrock
from transformers import BertForSequenceClassification, BertTokenizer
import re

llm = ChatBedrock(
    credentials_profile_name="sf-test", 
    model_id="us.anthropic.claude-3-sonnet-20240229-v1:0",
    temperature=0.5,
    max_tokens=2048
)

# Load and parse dataset
def load_kpi_edgar(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)

# LangChain prompt interface
def query_llm_with_langchain(company_name: str, year: str, text: str) -> List[Tuple[str, str, str]]:
    prompt = f"""
            You are a seasoned financial analyst, expert in financial statements review and extraction of KPI and its relations and actionable insight generation. 
            You are given a financial statement of {company_name} from a 10-K document filled with SEC in {year}. 
            Your task is to extract the key performance indicators (KPIs) and Genrate valuable and actionable insights for stakeholders (investors, auditors etc.) helping them in decision-making.
            Follow the below instructions carefully:
            1. Identify KPI names, which are financial terms representing the company's economic state (e.g., revenue, right-of-use, salary expenses, income tax etc.).
            2. Extract the values associated with each KPI, including currency symbols (e.g., $, €, ₹) and units (e.g., millions, billions) for understanding the correct scale.
            3. Classify the values into categories: "cy" (current year), "py" (previous year), and "py1" (year before the previous year) so that the comparisons could be drawn.
            4. Perform a SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis based on the extracted KPIs and their values.
            5. Summarize the analysis highlighting most important observations.

            Ensure -
            - The analysis must be factually correct and based only on the extracted tuples.
            - If a section of the SWOT analysis cannot be determined from the provided data, it should be excluded.
            - The analysis in one section must not contradict another section.
            - Do not include additional information from your knowledge.

            . 
            Provide analysis of this <financial_statement>"{text}"</financial_statement> based on provided instructions.
            The response must be formatted in a well-structured Markdown, with clear demarcation of each section using ##. 
            Sections to include are List of Key Performance Indicators, Strengths, Weaknesses, Opportunities, Threats and Summary
            """

    response = llm.invoke(prompt)
    return response.content

def get_company_name(ticker: str) -> str:
    ticker_file = "company_ticker_name.csv"
    ticker_data = pd.read_csv(ticker_file)
    company_name = ticker_data.loc[ticker_data['Ticker'] == ticker, 'Name']
    return company_name.values[0] if not company_name.empty else "Unknown Company"


def evaluate_factuality(text: str, gen: str) -> str:
    """
    Evaluate the factual correctness
    Returns "CORRECT" or "INCORRECT".
    """
    # Load FactCC model and split SWOT sections
    
    model_path = 'manueldeprada/FactCC'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    factcc_model = BertForSequenceClassification.from_pretrained(model_path)
    
    # Extract SWOT sections using regex
    strengths = re.search(r'##\s*Strengths\s*(.*?)(?=##|$)', gen, re.DOTALL)
    weaknesses = re.search(r'##\s*Weaknesses\s*(.*?)(?=##|$)', gen, re.DOTALL)
    opportunities = re.search(r'##\s*Opportunities\s*(.*?)(?=##|$)', gen, re.DOTALL)
    threats = re.search(r'##\s*Threats\s*(.*?)(?=##|$)', gen, re.DOTALL)
    
    sections = []
    for section in [strengths, weaknesses, opportunities, threats]:
        if section:
            sections.append(section.group(1).strip())
    
    # Evaluate each section
    results = []
    for section in sections:
        input_dict = tokenizer(text, section, max_length=512, padding='max_length', truncation='longest_first', return_tensors='pt')
        logits = factcc_model(**input_dict).logits
        pred = logits.argmax(dim=1)
        results.append(factcc_model.config.id2label[pred.item()])
    
    # Return CORRECT only if all sections are CORRECT
    return "CORRECT" if all(result == "CORRECT" for result in results) else "INCORRECT"

def evaluate_relevance_and_helpfulness(input_text: str, generation: str) -> dict:
    prompt = f"""
    You are an expert financial analyst judge. Evaluate the following generated analysis 
    for relevance and helpfulness. Score from 1-10 where 10 is best.
    **
    Input Text: {input_text}
    **
    Generated Analysis: {generation}
    **

    Rate on:
    1. Relevance - Is the analysis relevent to the input text and does not include additional information?
    2. Helpfulness - Does it provide easy to understand useful insights and explanations for stakeholders?

    Return only a numeric score between 1 and 10.
    """
    
    response = llm.invoke(prompt)
    try:
        score = json.loads(response.content)
        return score
    except:
        return 0

def run_langchain_kpi_eval(data: List[Dict[str, Any]], max_sentences: int = 50):
    results = []
    count = 0

    for doc in data:
        company_name = get_company_name(doc["id_"].split("_")[0])
        year = doc["id_"].split("_")[2].split("-")[1]
        for segment in doc["segments"]:
            text = segment["value"]
            gen = query_llm_with_langchain(company_name, text, year)

            file_name = f"C:/ProjectCodeBase/ms/insights2/{company_name}-{year}-{doc['id_']}-{segment['id_']}.md"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(f"# Financial Statement Analysis\n\n")
                f.write(f"## Text\n{text}\n\n")
                f.write(f"## Generated Analysis\n{gen}\n")
                f.write(f"## Factuality Evaluation\n{evaluate_factuality(text, gen)}\n")
                f.write(f"## Relavance Evaluation\n{evaluate_relevance_and_helpfulness(text, gen)}\n")

            results.append((text, gen))
            count += 1
            if count >= max_sentences:
                return results


if __name__ == "__main__":
    # evaluate_factuality('Earnings from Continuing Operations allocated to common shares in 2019 , 2018 and 2017 were $ 3.666 billion , $ 2.320 billion and $ 346 million , respectively.', '''[('earnings from continuing operations', '3.666 billion', 'cy'), ('earnings from continuing operations', '2.320 billion', 'py'), ('earnings from continuing operations', '346 million', 'py1')]''')

    load_dotenv()
    dataset = load_kpi_edgar("kpi-edgar\data\kpi_edgar.json")  
    results = run_langchain_kpi_eval(dataset, max_sentences=20)