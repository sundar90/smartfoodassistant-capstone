import gradio as gr
import pandas as pd
from io import BytesIO
from PIL import Image  # <-- Import PIL
from extract_nutrient import extraction_using_gemini
from extract_ingredient import FoodLabelExtractor
import tempfile
from standardise_nutrients import update_nutrient_dataframe
from predict_nutri_score import NutriScorePredictor
from nutriscore_explainability import get_nutriscore_insights

ingredient_extractor = FoodLabelExtractor()
nutriscore_predictor = NutriScorePredictor()

# Load your Nutri-Score images as PIL Images once at the top
def load_image(path):
    return Image.open(path).convert("RGB")

nutri_score_A_img = load_image("config/A.jpg")
nutri_score_B_img = load_image("config/B.jpg")
nutri_score_C_img = load_image("config/C.jpg")
nutri_score_D_img = load_image("config/D.jpg")
nutri_score_E_img = load_image("config/E.jpg")

def extract_nutrients(image):
    """
    Converts the uploaded PIL image to bytes (if necessary) and calls
    extract_nutrients_from_image (via Gemini) to extract nutrient data.
    Returns a Pandas DataFrame with nutrient information.
    """
    try:
        if hasattr(image, "save"):
            buf = BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        else:
            image_bytes = image

        df = extraction_using_gemini(image_bytes)
        return df
    except Exception as e:
        print("Error in nutrient extraction:", e)
        return None

def extract_ingredients_allergens(image):
    """
    Accepts an uploaded image (PIL Image, bytes, or file path), saves it if needed,
    and then extracts ingredients and corresponding allergens.
    Returns three outputs: ingredients_df, allergens_df, and impact.
    """
    try:
        if hasattr(image, "save"):
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image.save(temp_file, format="PNG")
            image_path = temp_file.name
            temp_file.close()
        elif isinstance(image, bytes):
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_file.write(image)
            temp_file.flush()
            image_path = temp_file.name
            temp_file.close()
        elif isinstance(image, str):
            image_path = image
        else:
            raise ValueError("Unsupported image format. Provide a PIL Image, bytes, or file path.")

        ingredients_df, allergens_df, impact = ingredient_extractor.process_image(image_path)
        return ingredients_df, allergens_df, impact
    except Exception as e:
        print("Error in ingredients/allergen extraction:", e)
        # Return empty/default values if extraction fails.
        return pd.DataFrame(), pd.DataFrame(), "Extraction error"

def extract_nutri_score(nutrient_df):
    """
    Predicts the Nutri-Score letter, then returns a PIL image object
    corresponding to that letter, plus the explanation.
    """
    try:
        if any(col.lower() == "nutrient" for col in nutrient_df.columns):
            nutrient_df = nutrient_df.rename(columns=lambda x: "Nutrient" if x.lower() == "nutrient" else x)

        modified_df = update_nutrient_dataframe(nutrient_df)
        nutri_score_letter = nutriscore_predictor.predict(modified_df)

        # Convert letter -> the loaded PIL image
        if nutri_score_letter == "A":
            nutri_score_image = nutri_score_A_img
        elif nutri_score_letter == "B":
            nutri_score_image = nutri_score_B_img
        elif nutri_score_letter == "C":
            nutri_score_image = nutri_score_C_img
        elif nutri_score_letter == "D":
            nutri_score_image = nutri_score_D_img
        elif nutri_score_letter == "E":
            nutri_score_image = nutri_score_E_img
        else:
            nutri_score_image = None

        nutrient_json = dict(zip(nutrient_df['Nutrient'], nutrient_df['per_100g']))
        nutrient_json.pop('energy_100g',None)
        explanation = get_nutriscore_insights(nutri_score_letter, nutrient_json)

        return nutri_score_image, explanation
    except Exception as e:
        print("Error in nutri score extraction:", e)
        return None, f"Error: {e}"

def clear_outputs(image):
    """
    Clears all UI elements when the image input is cleared.
    """
    empty_df = pd.DataFrame({"Message": [""]})
    cleared_allergen_table = gr.update(value=empty_df, visible=False)
    return (
        empty_df,  # nutrient_table
        "",        # ingredients_text_output
        "",        # impact_text_output
        None,      # allergen_state
        "",        # nutrient_message
        None,      # nutri_score_image (PIL)
        "Explanation: Not available",  # nutri_explanation_text
        cleared_allergen_table
    )

def process_image(image):
    """
    Processes the uploaded image to extract nutrient data, ingredients,
    and allergens. This generator function yields intermediate outputs so that
    nutrient extraction is displayed immediately and ingredients extraction follows.
    """
    if image is None:
        yield clear_outputs(image)
        return

    # Step 1: Nutrient extraction
    try:
        nutrient_df = extract_nutrients(image)
    except Exception as e:
        print("Error during nutrient extraction:", e)
        nutrient_df = None

    if nutrient_df is None:
        nutrient_message = "Automatic Extraction failed. Please create the table manually."
        nutrient_df = pd.DataFrame(columns=["nutrient", "per_100mg"])
    else:
        nutrient_message = ""

    # Compute nutrition analysis (PIL image + explanation)
    try:
        nutri_score_image, nutri_explanation = extract_nutri_score(nutrient_df)
    except Exception as e:
        print("Error computing nutrition analysis:", e)
        nutri_score_image, nutri_explanation = None, f"Error: {e}"

    yield (
        nutrient_df,
        "Extracting ingredients...",
        "Extracting allergen impact...",
        None,
        nutrient_message,
        nutri_score_image,
        nutri_explanation
    )

    # Step 2: Ingredients & allergens
    try:
        ingredients_df, allergens_df, impact = extract_ingredients_allergens(image)
    except Exception as e:
        print("Error during ingredients extraction:", e)
        ingredients_df, allergens_df, impact = pd.DataFrame(), pd.DataFrame(), "Extraction error"

    try:
        if not ingredients_df.empty:
            ingredients_text = ", ".join(ingredients_df.iloc[:, 0].astype(str).tolist())
        else:
            ingredients_text = "No ingredients found."
    except Exception as e:
        print("Error processing ingredients text:", e)
        ingredients_text = "No ingredients found."

    yield (
        nutrient_df,
        ingredients_text,
        impact,
        allergens_df,
        nutrient_message,
        nutri_score_image,
        nutri_explanation
    )

def show_allergens(allergen_data):
    return gr.update(value=allergen_data, visible=True)

def save_table_and_recalculate(edited_table):
    """
    Recalculates the nutrition score using the updated table,
    returns a PIL image + new explanation.
    """
    if not isinstance(edited_table, pd.DataFrame):
        try:
            edited_table = pd.DataFrame(edited_table, columns=["Nutrient", "Amount", "Unit"])
        except Exception as e:
            print("Error converting table:", e)

    print("Edited table:")
    print(edited_table)

    # Recalculate the nutrition score
    nutri_score_image, explanation = extract_nutri_score(edited_table)
    return (
        "Table saved successfully!",     # nutrient_message
        nutri_score_image,                 # nutri_score_image (PIL)
        explanation                        # nutri_explanation_text (markdown string)
    )

default_df = pd.DataFrame({"Message": ["upload an image to view extraction results"]})

css = """
    body {
        background: linear-gradient(to right, #ece9e6, #ffffff);
        font-family: 'Helvetica Neue', sans-serif;
        margin: 0;
        padding: 20px;
    }
    .header {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 30px;
        color: #333;
    }
    .upload-section {
        padding: 20px;
        border: 2px dashed #ccc;
        border-radius: 12px;
        background: #000;  /* change to black */
        color: #fff;       /* adjust text color */
        text-align: center;
    }
    .btn-primary {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 1rem;
        cursor: pointer;
        margin-top: 10px;
    }
    .btn-primary:hover {
        background-color: #45a049;
    }
    /* Uniform styling for all box-like elements */
    .dataframe-section, .card, .upload-section {
        background: #000;  /* set background to black */
        color: #fff;       /* text color white for contrast */
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 20px;
    }
    .gr-button, .gr-markdown, .gr-dataframe {
        background: inherit;
        color: inherit;
    }
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<div class='header'>Smart Food Assistant</div>")

    with gr.Row():
        # Left column: Upload image & Allergen Impact
        with gr.Column(scale=1, min_width=300):
            with gr.Column(elem_classes="upload-section"):
                gr.Markdown("### Upload Image")
                image_input = gr.Image(label="Upload Food Image", type="pil")
                submit_btn = gr.Button("Submit", elem_classes="btn-primary")

            with gr.Column(elem_classes="card"):
                gr.Markdown("### Allergen Impact")
                impact_text_output = gr.Markdown(value="No impact data yet.")
                click_more = gr.Button("Click More", elem_classes="btn-primary")
                allergen_table = gr.Dataframe(
                    value=default_df,
                    interactive=True,
                    label="Allergen Information",
                    visible=False
                )
                allergen_state = gr.State()

        # Right column: Nutrients, Ingredients, Nutri-Score
        with gr.Column(scale=1, min_width=400):
            # Nutrient Information
            with gr.Column(elem_classes="card"):
                gr.Markdown("### Nutrient Information")
                nutrient_message = gr.Markdown("")
                nutrient_table = gr.Dataframe(
                    value=default_df,
                    interactive=True,
                    label="Nutrient Information"
                )
                save_btn = gr.Button("Save Nutrient Table", elem_classes="btn-primary")

            # Ingredients
            with gr.Column(elem_classes="card"):
                gr.Markdown("### Ingredients")
                ingredients_text_output = gr.Markdown(value="No ingredients extracted yet.")

            # Nutrition Analysis (NutriScore as PIL image)
            with gr.Column(elem_classes="card"):
                gr.Markdown("### Nutrition Analysis")
                nutri_score_image = gr.Image(
                    label="Nutrition Score",
                    type="pil",
                    value=None
                )
                nutri_explanation_text = gr.Markdown("Explanation: Not available")

    # Process image on submit
    submit_btn.click(
        fn=process_image,
        inputs=[image_input],
        outputs=[
            nutrient_table,
            ingredients_text_output,
            impact_text_output,
            allergen_state,
            nutrient_message,
            nutri_score_image,
            nutri_explanation_text
        ]
    )

    # Clear outputs when image is cleared
    image_input.change(
        fn=clear_outputs,
        inputs=[image_input],
        outputs=[
            nutrient_table,
            ingredients_text_output,
            impact_text_output,
            allergen_state,
            nutrient_message,
            nutri_score_image,
            nutri_explanation_text,
            allergen_table
        ]
    )

    # Reveal allergen details
    click_more.click(
        fn=show_allergens,
        inputs=[allergen_state],
        outputs=[allergen_table]
    )

    # Recalculate Nutri-Score after manual table edits
    save_btn.click(
        fn=save_table_and_recalculate,
        inputs=[nutrient_table],
        outputs=[nutrient_message, nutri_score_image, nutri_explanation_text]
    )

demo.queue()
demo.launch(share=True)