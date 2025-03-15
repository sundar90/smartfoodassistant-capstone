import boto3
import os
import re
from textractor import Textractor
import textractor.data.constants
import pandas as pd
from rapidfuzz import process, fuzz
import tempfile

os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
os.environ["AWS_DEFAULT_REGION"] = aws_region

extractor = Textractor()

# Initialize Textract client
client = boto3.client('textract',aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"], region_name=os.environ["AWS_DEFAULT_REGION"])

with open('config/nutrients.txt', 'r') as nutrients:
    nutrient_values = nutrients.readlines()

nutrient_values = [line.strip() for line in nutrient_values]


def extract_tables(image_bytes):
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_file.flush()
            temp_file_path = temp_file.name

        # Now pass the temporary file path to the extractor
        document = extractor.analyze_document(
            temp_file_path,
            textractor.data.constants.TextractFeatures.TABLES
        )

        tables_list = []  # Store extracted tables as DataFrames

        for j, page in enumerate(document.pages):
            for i, table in enumerate(page.tables):
                df = pd.DataFrame(table.to_pandas())  # Convert table to DataFrame

                # Remove '\n' from all values in the DataFrame
                df = df.applymap(lambda x: x.replace("\n", " ") if isinstance(x, str) else x)

                tables_list.append(df)  # Append cleaned DataFrame to list

        return tables_list
    except Exception as e:
        return []

def extract_raw_text(image_bytes):
    # Write image bytes to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_file.write(image_bytes)
        temp_file.flush()
        temp_file_path = temp_file.name

    # Use Amazon Textract to detect text in the document
    document = extractor.detect_document_text(temp_file_path)

    # Collect all detected lines into a single string
    raw_text_lines = []
    for page in document.pages:
        for line in page.lines:
            raw_text_lines.append(line.text)

    # Optionally, join the lines with newlines
    return " ".join(raw_text_lines)

def clean_textract_nutrient_table(df, nutrient_list, match_threshold=70):
    """
    Cleans a Textract nutrient table.

    Assumptions & steps:
      1. The first row is the header.
      2. The first column is renamed to "nutrient_name".
         The 2nd and 3rd columns are checked: if the header contains "100" then rename to "per_100g",
         if it contains "30" then rename to "per_30g".
      3. For the "nutrient_name" column:
         - If nutrient quantity values are embedded (e.g. "Sugar 5g"), they are extracted to a new column
           "nutrient_quantity" (after cleaning).
         - The nutrient name is normalized and mapped to a standard nutrient name from nutrient_list.
           A manual mapping (e.g. mapping "of-which-total-sugar" to "sugars", or "energy" to None to drop)
           is applied first, then fuzzy matching is used.
      4. For all remaining quantity columns:
         - If percentage values are found (e.g. "10%"), they are extracted and removed from the cell.
           All extracted percentages are combined into a new column "percentage_daily_value".
         - Postprocessing on quantity values: remove any leading "<" and replace commas between digits with a period.

    Parameters:
      df (DataFrame): Raw DataFrame (with header from the first row).
      nutrient_list (list): List of standard nutrient names.
      match_threshold (int): Fuzzy matching threshold (0-100) for nutrient mapping.

    Returns:
      DataFrame: Cleaned DataFrame.
    """

    # ----------------------------
    # Helper Functions
    # ----------------------------

    def clean_quantity_value(text):
        """Remove any leading '<' and replace comma (with optional whitespace) between digits with a period."""
        text = text.lstrip('<').strip()
        text = re.sub(r'(\d+)[,]\s*(\d+)', r'\1.\2', text)
        return text

    # Pattern to capture a quantity (e.g. "5g", "18,4 mg", etc.)
    quantity_pattern = r'(\d+(?:[.,]\d+)?\s*(?:mg|g|kg|ml|l))'

    def extract_nutrient_quantity(text):
        """
        Extract a quantity from the nutrient name cell if present.
        Returns a tuple (cleaned_text, extracted_quantity).
        """
        match = re.search(quantity_pattern, text, flags=re.IGNORECASE)
        if match:
            qty = clean_quantity_value(match.group(1))
            # Remove the quantity from the nutrient text.
            new_text = re.sub(quantity_pattern, '', text, flags=re.IGNORECASE).strip()
            return new_text, qty
        return text, None

    # Manual mapping for nutrient names.
    manual_map = {
        "of-which-total-sugar": "sugars",
        "energy": None,  # Not a nutrient to include
        "protein": "proteins",
        "carbohydrate": "carbohydrates",
        "total-fat": "fat",
        "of-which": None
    }

    def standardize_nutrient(nutrient):
        """Normalize nutrient text and map it to a standard nutrient name."""
        normalized = nutrient.strip().lower().replace(" ", "-").replace('"', '').replace("'", "")
        if normalized in manual_map:
            return manual_map[normalized]
        match = process.extractOne(normalized, nutrient_list, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= match_threshold:
            return match[0]
        for standard in nutrient_list:
            if standard in normalized:
                return standard
        return None

        # ----------------------------
        # Step 1: Rename Columns
        # ----------------------------

        # Assume the DataFrame already has header names from the first row.

    def normalize_string(s):
        s = s.lower()
        s = re.sub(r'\s+', '', s)
        s = re.sub(r'[^\w]', '', s)
        return s

    df = df.copy()
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    print(df.head())
    cols = list(df.columns)
    # Rename first column to "nutrient_name"
    cols[0] = "nutrient_name"
    target = "per 100g"
    norm_target = normalize_string(target)
    # For 2nd and 3rd columns, if header text contains "100" or "30", rename accordingly.
    for i in range(1, len(cols)):
        header_val = str(cols[i])
        norm_header_val = normalize_string(header_val)
        # Use fuzzy matching to compare with the target.
        match = process.extractOne(norm_header_val, [norm_target], scorer=fuzz.ratio)
        if match[1] >= 80:  # Threshold can be adjusted as needed.
            cols[i] = "nutrient_quantity"
        elif "30" in header_val:
            cols[i] = "per_30g"
    print("colnames:", cols)
    print(df.head())
    df.columns = cols

    # ----------------------------
    # Step 2: Process the "nutrient_name" Column
    # ----------------------------

    nutrient_names = []
    embedded_nutrient_quantities = []  # Changed variable name to avoid conflict
    for val in df["nutrient_name"]:
        # Extract any embedded nutrient quantity.
        cleaned_val, qty = extract_nutrient_quantity(str(val))
        # Standardize the nutrient name.
        std_name = standardize_nutrient(cleaned_val)
        nutrient_names.append(std_name)
        embedded_nutrient_quantities.append(qty)

    df["nutrient_name"] = nutrient_names
    df["embedded_nutrient_quantity"] = embedded_nutrient_quantities
    # Drop rows where the nutrient name mapping failed (i.e. is None).
    df = df[df["nutrient_name"].notnull()].reset_index(drop=True)

    # ----------------------------
    # Step 3: Process Quantity Columns for Percentages & Clean Values
    # ----------------------------

    # Process all columns except "nutrient_name" and "nutrient_quantity".
    quantity_columns = [col for col in df.columns if col not in ["nutrient_name", "nutrient_quantity"]]

    # Prepare a new column for percentage daily values.
    percentage_daily_values = []

    for idx, row in df.iterrows():
        percentages = []
        for col in quantity_columns:
            cell = str(row[col])
            # Extract any percentage values, e.g. "10%" or "15,5 %"
            perc_matches = re.findall(r'(\d+(?:[.,]\d+)?\s*%)', cell)
            if perc_matches:
                for perc in perc_matches:
                    cleaned_perc = clean_quantity_value(perc)
                    percentages.append(cleaned_perc)
                # Remove percentage values from the cell.
                cell = re.sub(r'\d+(?:[.,]\d+)?\s*%', '', cell).strip()
            # Clean the remaining cell value.
            cell = clean_quantity_value(cell)
            df.at[idx, col] = cell
        percentage_daily_values.append(', '.join(percentages) if percentages else None)

    df["percentage_daily_value"] = percentage_daily_values

    return df

def extract_nutrients_from_image(img_bytes):
    # tables = extract_tables('actual_images/IMG20250219195332.jpg')
    try:
        tables = extract_tables(img_bytes)
        cleaned_df = clean_textract_nutrient_table(tables[0], nutrient_values)
        return cleaned_df
    except Exception as e:
        nutrients = [
            'energy_100g', 'fat_100g', 'saturated_fat_100g', 'trans-fat_100g',
            'cholesterol_100g', 'carbohydrates_100g', 'fiber_100g', 'proteins_100g',
            'salt_100g', 'sodium_100g', 'potassium_100g', 'sugars_100g',
            'calcium_100g', 'iron_100g', 'fruits-vegetables-nuts-estimate-from-ingredients_100g'
        ]

        df = pd.DataFrame({
            'Nutrient': nutrients,
            'per_100g': 0
        })
        return df




