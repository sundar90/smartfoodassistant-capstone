import os
import re
import json
import time
import cv2
import numpy as np
import boto3
from PIL import Image, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import google.generativeai as genai
from config.variables import GEMINI

class FoodLabelExtractor:
    def __init__(self):
        """
        Initialize API keys, credentials, and file paths.
        """
        # AWS credentials (for production, consider secure retrieval methods)

        # Gemini API key
        self.gemini_api_key = GEMINI.SECOND_KEY

        # File paths
        self.preprocessed_path = "config/preprocessed.png"
        self.allergens_path = "config/allergens.csv"

    def preprocess_image(self, image_path):
        """
        Preprocess the image for improved OCR performance.
        Resizes, converts to grayscale, removes noise, applies thresholding and sharpening.
        Displays the original and processed image side-by-side.
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print("Error: Unable to load image. Check file path.")
            return None

        steps = [("Original", image.copy())]

        # Resize for better OCR accuracy
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        steps.append(("Resized", image.copy()))

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        steps.append(("Grayscale", gray.copy()))


        # Contrast enhancement using histogram equalization
        equalized = cv2.equalizeHist(gray)
        steps.append(("Equalized", equalized.copy()))

        # Noise reduction with median blur
        blurred = cv2.medianBlur(equalized, 3)
        steps.append(("Noise Reduced", blurred.copy()))


        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 2)
        steps.append(("Thresholded", thresh.copy()))

        # Sharpen the image
        kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(thresh, -1, kernel_sharp)
        steps.append(("Sharpened", sharpened.copy()))

        # Save the processed image
        processed_path = self.preprocessed_path
        if not processed_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            processed_path += ".png"
        cv2.imwrite(processed_path, sharpened)

        # Display original and preprocessed images side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        display_steps = [("Original", image), ("Preprocessed", sharpened)]
        for ax, (title, img) in zip(axes, display_steps):
            if len(img.shape) == 3:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

        return sharpened

    def extract_text_with_textract(self, image_path, original_path):
        """
        Uses AWS Textract to extract text and layout information from an image.
        """
        textract = boto3.client('textract', region_name=self.aws_region,
                                aws_access_key_id=self.aws_access_key,
                                aws_secret_access_key=self.aws_secret_key,
                                )

        # Read image bytes
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()

        # Analyze document layout
        response = textract.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['LAYOUT']
        )

        # Check if required BlockType exists
        valid_block_types = {'LAYOUT_TITLE', 'LAYOUT_TEXT', 'LAYOUT_SECTION_HEADER',
                            'TABLE', 'TABLE_TITLE', 'LAYOUT_TABLE', 'LAYOUT_FIGURE',
                            'LAYOUT_FOOTER'}

        found_valid_block = False
        found_ingredient_text = False

        for block in response.get('Blocks', []):
            if block.get('BlockType') in valid_block_types:
                found_valid_block = True

            if block.get('BlockType') == 'LINE' and 'Text' in block:
                if 'ingredient' in block['Text'].lower():
                    found_ingredient_text = True

        # If no valid block types or 'ingredient' keyword found, retry with original_path
        if not found_valid_block or not found_ingredient_text:
            #print("Reprocessing with original_path...")
            with open(original_path, 'rb') as image_file:
                image_bytes = image_file.read()

            # Analyze document layout
            response = textract.analyze_document(
                Document={'Bytes': image_bytes},
                FeatureTypes=['LAYOUT']
            )
            return self.process_layout_blocks(response, original_path)

        return self.process_layout_blocks(response, image_path)

    def process_layout_blocks(self, response, image_path):
        """
        Processes Textract layout blocks, draws bounding boxes on the image,
        and returns a DataFrame mapping block types to extracted text.
        """
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        width, height = image.size

        layout_blocks = []
        line_blocks = []

        # Separate layout blocks and line blocks
        for block in response.get('Blocks', []):
            if block['BlockType'] in ['LAYOUT_TITLE', 'LAYOUT_TEXT', 'LAYOUT_SECTION_HEADER',
                                      'TABLE', 'TABLE_TITLE', 'LAYOUT_TABLE', 'LAYOUT_FIGURE',
                                      'LAYOUT_FOOTER']:
                layout_blocks.append(block)
            elif block['BlockType'] == 'LINE':
                line_blocks.append(block)

        layout_text_mapping = []
        i = 0
        while i < len(layout_blocks):
            layout_block = layout_blocks[i]
            layout_bbox = layout_block['Geometry']['BoundingBox']
            merged_text = []
            block_type = layout_block['BlockType']

            # Merge section header with the next block if applicable
            if block_type == 'LAYOUT_SECTION_HEADER' and i + 1 < len(layout_blocks):
                next_block = layout_blocks[i + 1]
                if next_block['BlockType'] in ['LAYOUT_TEXT', 'TABLE', 'LAYOUT_TABLE', 'LAYOUT_FIGURE']:
                    next_bbox = next_block['Geometry']['BoundingBox']
                    layout_bbox = {
                        'Left': min(layout_bbox['Left'], next_bbox['Left']),
                        'Top': min(layout_bbox['Top'], next_bbox['Top']),
                        'Width': max(layout_bbox['Left'] + layout_bbox['Width'],
                                     next_bbox['Left'] + next_bbox['Width']) - min(layout_bbox['Left'], next_bbox['Left']),
                        'Height': max(layout_bbox['Top'] + layout_bbox['Height'],
                                      next_bbox['Top'] + next_bbox['Height']) - min(layout_bbox['Top'], next_bbox['Top'])
                    }
                    block_type = f"{layout_block['BlockType']} + {next_block['BlockType']}"
                    i += 1  # Skip the next block since it is merged

            for line_block in line_blocks:
                if self.is_inside(line_block['Geometry']['BoundingBox'], layout_bbox):
                    merged_text.append(line_block.get('Text', ''))
            merged_text = " ".join(merged_text)
            layout_text_mapping.append({
                "Block Type": block_type,
                "Extracted Text": merged_text
            })

            # Draw bounding box for visualization
            left = layout_bbox['Left'] * width
            top = layout_bbox['Top'] * height
            right = left + (layout_bbox['Width'] * width)
            bottom = top + (layout_bbox['Height'] * height)
            color_map = {
                'LAYOUT_TITLE': "blue",
                'LAYOUT_SECTION_HEADER': "orange",
                'LAYOUT_FIGURE': "black",
                'LAYOUT_TEXT': "green",
                'LAYOUT_TABLE': "grey",
            }
            color = color_map.get(layout_block['BlockType'], "red")
            draw.rectangle([left, top, right, bottom], outline=color, width=2)

            i += 1

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        return pd.DataFrame(layout_text_mapping)

    @staticmethod
    def is_inside(bbox_small, bbox_large):
        """
        Check if a small bounding box is inside a larger bounding box.
        """
        return (
            bbox_small['Left'] >= bbox_large['Left'] and
            bbox_small['Top'] >= bbox_large['Top'] and
            bbox_small['Left'] + bbox_small['Width'] <= bbox_large['Left'] + bbox_large['Width'] and
            bbox_small['Top'] + bbox_small['Height'] <= bbox_large['Top'] + bbox_large['Height']
        )

    @staticmethod
    def filter_ingredients_and_allergens(df):
        """
        Filter DataFrame rows to those containing the words 'ingredient' or 'allergen'.
        """
           # List of translations for 'ingredient' and 'allergen' in multiple languages
        terms = [
            'ingredient', 'allergen',  # English
            'ingrédient', 'allergène',  # French
            'ingrediente', 'alérgeno',  # Spanish
            'Zutat', 'Allergen',  # German
            'ingrediente', 'alérgeno',  # Italian
            'ingrediente', 'alérgeno',  # Portuguese
            '成分', 'アレルゲン',  # Japanese
            '성분', '알레르겐',  # Korean
            '成分', '过敏原',  # Chinese (Simplified)
            'bestanddeel', 'allergeen',  # Dutch
            'состав', 'аллерген',  # Russian
            'malzeme', 'alerjen',  # Turkish
            'ingrediënt', 'allergeen',  # Dutch
            'ingrediente', 'alérgeno',  # Romanian
            'ingrediente', 'alérgeno',  # Galician
            'ingrediente', 'alérgeno',  # Catalan
            'ingrediente', 'alérgeno',  # Basque
            'ingrediente', 'alérgeno',  # Filipino
            'bahan', 'alergen',  # Indonesian
            'bahan', 'alergen',  # Malay
            'ส่วนประกอบ', 'สารก่อภูมิแพ้',  # Thai
            'thành phần', 'chất gây dị ứng',  # Vietnamese
            'عنصر', 'مسبب للحساسية',  # Arabic
            'ingrediens', 'allergeen',  # Afrikaans
            'komponent', 'alergen',  # Estonian
            'ainesosa', 'allergeeni',  # Finnish
            'συστατικό', 'αλλεργιογόνο',  # Greek
            'חומר', 'אלרגן',  # Hebrew
            'összetevő', 'allergén',  # Hungarian
            'innihaldsefni', 'ofnæmisvaldur',  # Icelandic
            'bestanddeel', 'allergeen',  # Dutch
            'składnik', 'alergen',  # Polish
            'ingrediente', 'alérgeno',  # Portuguese
            'ingredientă', 'alergen',  # Romanian
            'ингредиент', 'аллерген',  # Russian
            'zložka', 'alergén',  # Slovak
            'sestavina', 'alergen',  # Slovenian
            'ingrediente', 'alérgeno',  # Spanish
            'ingrediens', 'allergen',  # Swedish
            'malzeme', 'alerjen',  # Turkish
            'інгредієнт', 'алерген',  # Ukrainian
            'thành phần', 'chất gây dị ứng'  # Vietnamese
        ]
        pattern = re.compile(r'\b(?:' + '|'.join(terms) + r')\w*\b', re.IGNORECASE)
        return df[df['Extracted Text'].str.contains(pattern, na=False)]

    @staticmethod
    def process_ingredients_regexp(df_filtered):
        """
        Process ingredients using regular expressions.
        """
        formatted_data = []
        for text in df_filtered['Extracted Text']:
            text = re.sub(r'(?i)^(ingredients|allergens)[:\-]?\s*', '', text).strip()
            main_ingredients = re.split(r',\s*(?![^()]*\))|[.;:]\s*|\band\b', text)
            for ingredient in main_ingredients:
                ingredient = ingredient.strip()
                match = re.match(r'([^()]+)\s*\(([^)]+)\)', ingredient)
                if match:
                    main = match.group(1).strip()
                    sub_items = match.group(2).strip()
                    formatted_data.append(f"{main} - {sub_items}")
                else:
                    formatted_data.append(ingredient)
        # Remove duplicates while preserving order
        formatted_data = list(dict.fromkeys(formatted_data))
        return pd.DataFrame({"Ingredient": formatted_data})

    def process_ingredients_gemini(self, extracted_text, batch_size=3):
        """
        Process ingredients using the Gemini generative API.
        """
        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

        prompt = f"""
        Extract only the ingredient names from the following food label text.
        Break down compound ingredients into their individual components.
        Make them singular and in raw form.

        **Food Label Text:**
        {json.dumps(extracted_text)}

        **Return ingredient names separated by a new line character**
        """
        try:
            time.sleep(2)  # Prevents rate limit issues
            response = model.generate_content(prompt)
            if response.text:
                return response.text
        except json.JSONDecodeError:
            print("Error: Gemini returned invalid JSON. Check response format.")
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
        return ""

    def extract_ingredients(self, df_filtered):
        """
        Extract ingredients from filtered text using Gemini; fallback to regex if needed.
        """
        merged_text = " ".join(df_filtered["Extracted Text"].dropna().tolist()).strip()
        result = self.process_ingredients_gemini(merged_text)
        ingredients_list = [line.replace("-", "").strip() for line in result.split("\n") if line.strip()]
        extracted_ingredients = pd.DataFrame(ingredients_list, columns=["Ingredient"])
        if extracted_ingredients.empty:
            extracted_ingredients = self.process_ingredients_regexp(df_filtered)
        return extracted_ingredients

    @staticmethod
    def clean_ingredient(ingredient):
        """
        Clean ingredient string by lowering case and removing parentheses content.
        """
        ingredient = ingredient.lower().strip()
        if "cocoa" in ingredient or "cacao" in ingredient:
            return "cacao bean"
        if "water" in ingredient and "melon" not in ingredient:
            return "water"
        return re.sub(r"\(.*?\)", "", ingredient).strip().lower()

    def find_best_match_tfidf(self, ingredient, vectorizer, vectors, choices, allergen_map, impact_map, category_map, threshold=0.7):
        """
        Use TF-IDF to find the best allergen match for an ingredient.
        """
        ingredient_clean = self.clean_ingredient(ingredient)
        ingredient_vector = vectorizer.transform([ingredient_clean])
        similarities = cosine_similarity(ingredient_vector, vectors).flatten()
        best_match_idx = similarities.argmax()
        best_match_score = similarities[best_match_idx]
        if best_match_score > threshold:
            best_match = choices[best_match_idx]
            return best_match, allergen_map.get(best_match, "None"), impact_map.get(best_match, "None"), category_map.get(best_match, "None")
        return ingredient_clean, "None", "None", "None"

    def extract_allergens(self, extracted_ingredients):
        """
        Extract allergen information by matching ingredients against an allergens CSV.
        """
        df_allergens = pd.read_csv(self.allergens_path)
        df_allergens.columns = df_allergens.columns.str.strip()
        extracted_ingredients.columns = extracted_ingredients.columns.str.strip()

        allergen_mapping = dict(zip(df_allergens["Food"].str.lower(), df_allergens["Allergy"]))
        impact_mapping = dict(zip(df_allergens["Food"].str.lower(), df_allergens["Impact"]))
        category_mapping = dict(zip(df_allergens["Food"].str.lower(), df_allergens["Category"]))
        allergen_foods = [self.clean_ingredient(food) for food in df_allergens["Food"]]

        vectorizer_allergen = TfidfVectorizer().fit(allergen_foods)
        allergen_vectors = vectorizer_allergen.transform(allergen_foods)

        matches = extracted_ingredients["Ingredient"].apply(
            lambda x: self.find_best_match_tfidf(x, vectorizer_allergen, allergen_vectors, allergen_foods,
                                                 allergen_mapping, impact_mapping, category_mapping)
        )
        matches_df = pd.DataFrame(matches.tolist(), columns=["Allergy Ingredient", "Allergy", "Impact", "Category"],
                                  index=extracted_ingredients.index)
        return extracted_ingredients.join(matches_df)

    @staticmethod
    def get_severity(allergy_category):
        """
        Determine the severity rank based on the allergy category.
        """
        severity_order = ["severe", "artificial", "conditions", "moderate", "mild"]
        for severity in severity_order:
            if severity in str(allergy_category).lower():
                return severity_order.index(severity)
        return len(severity_order)

    @staticmethod
    def display_result(extracted_ingredients, impact):
        """
        Display the final ingredients and allergen tables.
        """
        extracted_ingredients["Ingredient"] = extracted_ingredients["Ingredient"].replace("None", "").str.replace("(", "-", regex=False)
        filtered_ingredients = extracted_ingredients[["Ingredient"]].drop_duplicates()

        ingredients_table = [[row["Ingredient"]] for _, row in filtered_ingredients.iterrows() if row["Ingredient"]]
        print("\n\033[1m🍽️ Ingredients List:\033[0m")
        print("-" * 40)
        print(tabulate(ingredients_table, headers=["Ingredient"], tablefmt="grid"))

        severity_colors = {
            "severe": "\033[91m",       # Red
            "artificial": "\033[33m",   # Dark Orange
            "conditions": "\033[93m",   # Light Orange
            "moderate": "\033[94m",     # Cyan
            "mild": "\033[92m"          # Green
        }

        allergen_rows = extracted_ingredients[
            (extracted_ingredients["Allergy"].notna()) & (extracted_ingredients["Allergy"] != "None") &
            (extracted_ingredients["Allergy Ingredient"].notna()) & (extracted_ingredients["Allergy Ingredient"] != "None")
        ].drop_duplicates(subset=["Allergy Ingredient"])

        if not allergen_rows.empty:
            allergen_rows["Severity Rank"] = allergen_rows["Category"].apply(FoodLabelExtractor.get_severity)
            allergen_rows = allergen_rows.sort_values("Severity Rank")
            allergens_table = []
            for _, row in allergen_rows.iterrows():
                category = row["Category"]
                color = ""
                for key in severity_colors:
                    if key in str(category).lower():
                        color = severity_colors[key]
                        break
                allergens_table.append([
                    f"{color}{row['Category']}\033[0m",
                    f"{color}{row['Allergy Ingredient']}\033[0m",
                    f"{color}{row['Allergy']}\033[0m",
                    f"{color}{row['Impact']}\033[0m"
                ])
            print("\n\033[1m⚠️  Allergens Detected:\033[0m")
            print("-" * 40)
            print(tabulate(allergens_table, headers=["Category", "Allergy Ingredient", "Allergy", "Impact"], tablefmt="grid"))
        else:
            print("\n\033[1m⚠️  No allergens detected!\033[0m")
            print("-" * 40)
        if impact:
            print("Product Health Impact")
            print(impact)
            print("-" * 40)


    def impact_analysis_gemini(self, df):
        # Load API Key
        genai.configure(api_key=self.gemini_api_key)

        # Load Gemini Model
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

        # Convert DataFrame to Markdown Table
        df_markdown = df[['Ingredient', 'Allergy', 'Category']].to_markdown()

        # Thoughtful Prompt for Gemini
        prompt = f"""
        Analyze the overall ingredient and allergen impact of the following ingredients:
        {df_markdown}
        Provide a balanced summary in 3 bullet points covering:
        Potential allergic reactions and severity and Cumulative effects if any
        Overall healthiness for non-allergic individuals
        Final verdict on product suitability (balanced view on allergic: risks vs. benefits).
        Overall, Make it in 40 words.
        """
        # Call Gemini API
        response = model.generate_content(prompt)
        return response.text


    def process_image(self, image_path):
        """
        Main processing function. Validates the image file, preprocesses it,
        extracts text using Textract, filters for ingredients/allergens, and finally
        extracts and displays the ingredient and allergen information.
        """
        if not os.path.exists(image_path):
            print("Image file does not exist!")
            return

        # Preprocess the image and save the result
        preprocessed_image = self.preprocess_image(image_path)
        if preprocessed_image is None:
            return

        # Use the preprocessed image for Textract extraction
        processed_path = self.preprocessed_path
        df = self.extract_text_with_textract(processed_path,image_path)

        # Filter text blocks for ingredients and allergens
        if df.empty:
          print("No ingredients detected")
          return
        else:
            # Filter ingredients and allergens
            df_filtered = self.filter_ingredients_and_allergens(df)

            if df_filtered.empty:
                print("No ingredients detected")
                return
            else:
                # Extract ingredients
                extracted_ingredients = self.extract_ingredients(df_filtered)

                if extracted_ingredients.empty:
                    print("No ingredients detected")
                    return
                else:
                # Extract allergens
                    extracted_ingredients_allergens = self.extract_allergens(extracted_ingredients)
                    if extracted_ingredients_allergens.empty:
                        print("No ingredients detected")
                        return
                    else:
                        # Display final result
                        impact = self.impact_analysis_gemini(extracted_ingredients_allergens)
                        self.display_result(extracted_ingredients_allergens,impact)

        return extracted_ingredients, extracted_ingredients_allergens, impact

if __name__ == "__main__":
    import os
    os.chdir("C:/Users/rampa/Downloads/northwestern/northwestern/capstone_project/final_code")
    extractor = FoodLabelExtractor()
    image_path = "C:/Users/rampa/Downloads/northwestern/northwestern/capstone_project/image_extraction/actual_images/image_manju_2.jpg"
    extractor.process_image(image_path)