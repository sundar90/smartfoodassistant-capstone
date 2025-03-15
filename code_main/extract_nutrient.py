import requests
import json
import glob
import os
import io
import re
from PIL import Image
import pandas as pd
from typing import List
from pydantic import BaseModel
from config.variables import GEMINI
from extract_text_using_Rekognition import extract_text_with_rekognition
from nutrient_extraction_textract import extract_nutrients_from_image

# Define Pydantic models for the expected response.
class NutrientEntry(BaseModel):
    Nutrient: str
    per_100g: float

class NutrientsTable(BaseModel):
    data: List[NutrientEntry]

api_key = GEMINI.KEY

def clean_response_text(text: str) -> str:
    """
    Remove markdown code block formatting (```json ... ```) from the text.
    """
    # Remove triple backticks and optional language specifier if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def process_nutrient_value(value_str: str) -> float:
    """
    Process a nutrient value string by:
      - Removing units.
      - Converting mg to g.
      - For energy values: if both kJ and Cal/kcal are present, returns the kcal value;
        if only kJ is present, converts kJ to kcal.
    """
    value_str = value_str.strip()

    # Handle energy values: if a calorie value is present in parentheses, use that.
    cal_match = re.search(r"\(([\d\.]+)\s*(?:Cal|kcal)\)", value_str, re.IGNORECASE)
    if cal_match:
        return round(float(cal_match.group(1)),2)

    # If no calorie value, but kJ is present, convert kJ to kcal.
    kj_match = re.search(r"([\d\.]+)\s*kJ", value_str, re.IGNORECASE)
    if kj_match:
        kj_value = float(kj_match.group(1))
        # 1 kcal ≈ 4.184 kJ
        return round(kj_value / 4.184,2)

    # Convert mg to g.
    mg_match = re.search(r"([\d\.]+)\s*mg", value_str, re.IGNORECASE)
    if mg_match:
        mg_value = float(mg_match.group(1))
        return round(mg_value / 1000.0,2)

    # Assume grams if "g" is present.
    g_match = re.search(r"([\d\.]+)\s*g", value_str, re.IGNORECASE)
    if g_match:
        return round(float(g_match.group(1)),2)

    # Fallback: extract any number that might be present.
    num_match = re.search(r"([\d\.]+)", value_str)
    if num_match:
        return round(float(num_match.group(1)),2)

    raise ValueError(f"Could not process nutrient value: {value_str}")


def clean_nutrients_json(nutrients_json: dict) -> dict:
    """
    Cleans nutrient data from the input dictionary.

    For each nutrient entry:
      - Processes the `per_100g` field by removing units.
      - Converts mg values to g.
      - For energy values, retains only the kcal value (or converts kJ to kcal if needed).

    Args:
        nutrients_json (dict): A dictionary containing nutrient data with the key "data".

    Returns:
        dict: A new dictionary with the processed nutrient values.
    """
    processed_entries = []
    for entry in nutrients_json.get("data", []):
        nutrient = entry.get("Nutrient")
        value_str = entry.get("per_100g", "")
        try:
            processed_value = process_nutrient_value(value_str)
        except ValueError as e:
            # Here you can choose to log, skip, or set a default value. For now, we set to None.
            processed_value = None

        processed_entries.append({
            "Nutrient": nutrient,
            "per_100g": processed_value
        })

    return {"data": processed_entries}

def extract_nutrients_as_dataframe(df, ocr_text: str) -> pd.DataFrame:
    """
    Extracts nutritional information from OCR text using the Gemini API and returns it as a pandas DataFrame.

    Args:
        ocr_text (str): The OCR-extracted text containing nutritional information.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted nutritional information.
    """
    # Gemini API endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    # Create a prompt to extract nutrient information and return it as JSON that conforms to our schema.
    prompt = f"""
Extract the nutrients information table from the OCR text provided below.  Once you extract correct the values (order of values) using the tabular output provided.
Return your answer as a JSON object with a single key "data", whose value is a list of rows.
Each row must be an object with exactly two keys:
    - "Nutrient": the nutrient name (convert to English if it is in other language)
    - "per_100g": the numeric value for the nutrient per 100g (extract with unit. if multiple values exist, choose the one with KJ unit).
Do not include any additional text or formatting; output only the JSON.

OCR Text:
{ocr_text}
Tabular output:
{df}
    """

    # Create the payload as expected by the Gemini API.
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    # Set the header for JSON content.
    headers = {
        "Content-Type": "application/json"
    }

    # Send the POST request to the Gemini API.
    response = requests.post(url, headers=headers, json=payload)

    # Check if the request was successful.
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    # Parse the JSON response from the API.
    result = response.json()

    # Extract the response text containing the JSON.
    content_text = result['candidates'][0]['content']['parts'][0]['text']

    # Clean the response to remove any markdown formatting.
    content_text = clean_response_text(content_text)

    # Parse the JSON string into a Python dictionary.
    try:
        data_json = json.loads(content_text)
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON from response: {e}")

    cleaned_data = clean_nutrients_json(data_json)
    # Use Pydantic to enforce the expected schema.
    table = NutrientsTable(**cleaned_data)

    # Convert the list of nutrient entries to a DataFrame.
    df = pd.DataFrame([entry.dict() for entry in table.data])
    return df

def extraction_using_gemini(image_bytes: bytes) -> pd.DataFrame:
    ocr_text = extract_text_with_rekognition(image_bytes)
    # df = extract_nutrients_from_image(image_bytes)
    df=pd.DataFrame()
    nutrients_df = extract_nutrients_as_dataframe(df,ocr_text)
    return nutrients_df

# Example usage
if __name__ == "__main__":
    # Directory containing the images.
    image_file = "C:/Users/rampa/Downloads/northwestern/northwestern/capstone_project/image_extraction/downloaded_images_2/image_1432578.jpg"
    with Image.open(image_file) as img:
        # Display the image in a window
        img.show()
        img_bytes_io = io.BytesIO()
        img.save(img_bytes_io, format=img.format)
        image_bytes = img_bytes_io.getvalue()
    df = extraction_using_gemini(image_bytes)
    print(f"Extracted Nutrients DataFrame for {os.path.basename(image_path)}:")
    print(df)