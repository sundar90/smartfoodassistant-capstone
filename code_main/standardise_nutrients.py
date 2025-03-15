import re
import ast
import requests
import pandas as pd
from config.variables import GEMINI

# Define the standardized nutrients list.
standardised_nutrients = [
    'energy_100g', 'fat_100g', 'saturated-fat_100g', 'trans-fat_100g',
    'cholesterol_100g', 'carbohydrates_100g', 'fiber_100g', 'proteins_100g',
    'salt_100g', 'sodium_100g', 'potassium_100g', 'sugars_100g',
    'calcium_100g', 'iron_100g', 'fruits-vegetables-nuts-estimate-from-ingredients_100g'
]


def get_mapped_nutrients(nutrient_list):
    """
    Calls the Gemini LLM API to map the provided nutrient list to the corresponding standardized names.

    Args:
        nutrient_list (list): List of nutrient names (e.g., ["Energy", "Fat", ...]).

    Returns:
        list: The mapped nutrient names as returned by the API.
    """
    # Retrieve API key and build the API URL.
    api_key = GEMINI.KEY
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    # Create the prompt for the API.
    prompt = (
        f"map the nutrients in the inout nutrient list {nutrient_list} "
        f"to the corresponding names in {standardised_nutrients} wherever there is a match. "
        "return the response as a python array. do not provide any ptyhon code. provide the answer directly"
    )

    # Build the payload.
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
    result = response.json()

    # Extract the answer text from the API response.
    code_snippet = result['candidates'][0]['content']['parts'][0]['text']

    # Use a regular expression to extract the Python list from the response.
    extracted_list = ast.literal_eval(re.search(r'\[.*?\]', code_snippet).group())
    return extracted_list


def update_nutrient_dataframe(df):
    """
    Updates a DataFrame by mapping the nutrient names in the 'nutrient' column to standardized names
    using the Gemini API. The DataFrame must have columns ["nutrient", "per_100g"].

    Args:
        df (pd.DataFrame): Input DataFrame with columns "nutrient" and "per_100g".

    Returns:
        pd.DataFrame: The DataFrame with the 'nutrient' column updated to the mapped names.
    """
    # Extract the nutrient names from the DataFrame.
    nutrient_list = df['Nutrient'].tolist()
    # Get the mapped nutrient names using the Gemini API.
    mapped_nutrients = get_mapped_nutrients(nutrient_list)
    # Update the DataFrame.
    df['Nutrient'] = mapped_nutrients
    return df


# -----------------------------
# Example usage in a Jupyter Notebook cell:
# -----------------------------
# Create a sample DataFrame with columns ["nutrient", "per_100g"].
data = {
    'Nutrient': ["Energy", "Fat", "Saturated Fat", "Carbohydrate", "Sugar", "Fiber", "Protein", "Salt"],
    'per_100g': [31, 0, 0, 7.5, 7.4, 0.5, 2.5, 0.07]
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Update the DataFrame with the mapped nutrient names.
updated_df = update_nutrient_dataframe(df)

print("\nUpdated DataFrame:")
print(updated_df)
