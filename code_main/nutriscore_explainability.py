import requests
from pydantic import BaseModel, ValidationError
from config.variables import GEMINI
# Store API key securely (Set in environment variables for security)
API_KEY = GEMINI.KEY  # Replace with your actual key if needed
import json

# Function to analyze NutriScore impact (old version omitted for brevity)

class GeminiResponse(BaseModel):
    improvement_table: str

def get_nutriscore_insights(nutriscore_grade, nutrient_values):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI.KEY}"

    # Convert nutrient_values dict to a concise string:
    nutrient_text = ", ".join([f"{key}: {value}g" for key, value in nutrient_values.items()])

    # Updated prompt with new table headers and recommendation values
    prompt = f"""
Analyze the relationship between the predicted NutriScore grade and the provided nutrient values.

Input:
- Predicted NutriScore Grade: {nutriscore_grade}
- Nutrient Values: {nutrient_text}

Task:
Suggest a minimal change in the nutrients (except change in energy) which will improve the NutriScore by one step.
Include a table wiuth similar structure with different values as per input nutrients, verbatim, in your final answer:

| **Reason for this score**     | **Desired Nutrient Levels**                                                          |
|-------------------------------|-----------------------------------------------------------------------------------------|
| **Energy (494 kcal/100 g)**   | Consider products with around **400 kcal/100 g** to lower overall energy density.       |
| **Protein (6.6 g/100 g)**     | Consider products with around **8 g/100 g** to achieve a balanced protein content.      |

Return only valid JSON using the following schema:

{{
  "improvement_table": "Above table in valid Markdown"
}}

Important: Do not include any additional keys, text, or formatting outside this JSON.
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}

    try:
        # Make the request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        # Extract the raw text from Gemini's response
        response_text = (
            result.get("candidates", [{}])[0]
                 .get("content", {})
                 .get("parts", [{}])[0]
                 .get("text", "")
        )
        lines = response_text.strip().splitlines()
        # The JSON content is between the first and last lines (the fences)
        json_str = "\n".join(lines[1:-1])

        # Parse the cleaned JSON string into a Python dictionary
        data = json.loads(json_str)

        # Extract the value associated with "improvement_table"
        improvement_table = data["improvement_table"]
        # Use Pydantic to parse the JSON string into our GeminiResponse model
        parsed = GeminiResponse.parse_raw(json_str)

        # Return just the table as a string
        return parsed.improvement_table

    except ValidationError as ve:
        # If the JSON is malformed or missing the required key
        return f"Parsing error: {ve}"
    except Exception as e:
        # For other request/connection errors
        return f"Request error: {e}"


# Example usage
if __name__ == "__main__":
    # Sample input data
    nutriscore_grade = "D"
    nutrient_values = {
        "fat_100g": 2.2,
        "saturated-fat_100g": 0.3,
        "sugars_100g": 0.3,
        "fiber_100g": 7,
        "proteins_100g": 6.6,
        "salt_100g": 0
    }
    ingredients = ['chickpea', 'water', 'salt']

    # Get NutriScore insights
    insights = get_nutriscore_insights(nutriscore_grade, nutrient_values)

    # Print results
    print("\n🔹 *NutriScore Analysis Report* 🔹")
    print(insights)
