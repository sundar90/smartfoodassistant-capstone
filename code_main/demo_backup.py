import gradio as gr
import pandas as pd
from io import BytesIO
from final_code.nutrient_extraction_textract import extract_nutrients_from_image
from final_code.extract_nutrient import extraction_using_gemini
from final_code.extract_ingredient_final import FoodLabelExtractor
import tempfile
from final_code.standardise_nutrients import update_nutrient_dataframe
from final_code.predict_nutri_score import NutriScorePredictor

ingredient_extractor = FoodLabelExtractor()
nutriscore_predictor = NutriScorePredictor()


def extract_nutrients(image):
    """
    Converts the uploaded PIL image to bytes (if necessary) and calls
    extract_nutrients_from_image (via Gemini) to extract nutrient data.
    Returns a Pandas DataFrame with nutrient information.
    """
    if hasattr(image, "save"):
        buf = BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
    else:
        image_bytes = image

    df = extraction_using_gemini(image_bytes)
    return df


def extract_ingredients_allergens(image):
    """
    Accepts an uploaded image (PIL Image, bytes, or file path), saves it if needed,
    and then extracts ingredients and corresponding allergens.
    Returns three outputs: ingredients_df, allergens_df, and impact.
    """
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

def extract_nutri_score(nutrient_df):
    modified_df = update_nutrient_dataframe(nutrient_df)
    nutri_score,explanation = nutriscore_predictor.predict(modified_df)
    return nutri_score, explanation

def process_image(image):
    """
    Processes the uploaded image to extract nutrient data, ingredients,
    and allergens. This generator function yields intermediate outputs so that
    nutrient extraction is displayed immediately and ingredients extraction follows.
    """
    # Step 1: Nutrient extraction
    nutrient_df = extract_nutrients(image)
    if nutrient_df is None:
        nutrient_message = "Automatic Extraction failed. Please create the table manually."
        nutrient_df = pd.DataFrame(columns=["nutrient", "per_100mg"])
    else:
        nutrient_message = ""

    # Yield nutrient results immediately along with placeholder texts for other outputs.
    # Outputs order: nutrient_table, ingredients_text_output, impact_text_output, allergen_state, nutrient_message
    yield nutrient_df, "Extracting ingredients...", "Extracting allergen impact...", None, nutrient_message

    # Step 2: Ingredients and allergen extraction
    ingredients_df, allergens_df, impact = extract_ingredients_allergens(image)
    if not ingredients_df.empty:
        ingredients_text = ", ".join(ingredients_df.iloc[:, 0].astype(str).tolist())
    else:
        ingredients_text = "No ingredients found."

    # Yield final results
    yield nutrient_df, ingredients_text, impact, allergens_df, nutrient_message


def show_allergens(allergen_data):
    """
    Updates the allergens table component to make it visible and load the allergen data.
    """
    return gr.update(value=allergen_data, visible=True)


def save_table(edited_table):
    """
    Receives the edited nutrient table, prints it, and returns a status message.
    """
    if not isinstance(edited_table, pd.DataFrame):
        try:
            edited_table = pd.DataFrame(edited_table, columns=["Nutrient", "Amount", "Unit"])
        except Exception as e:
            print("Error converting table:", e)
    print("Edited table:")
    print(edited_table)
    return "Table saved successfully!"


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
        background: #f9f9f9;
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
    .dataframe-section {
        padding: 20px;
        background: #fefefe;
        border: 1px solid #ddd;
        border-radius: 12px;
    }
    .card {
        padding: 20px;
        background: #fefefe;
        border: 1px solid #ddd;
        border-radius: 12px;
        margin-top: 20px;
    }
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<div class='header'>Smart Food Assistant</div>")

    with gr.Row():
        # Left column: Upload image and Allergen Impact under the preview
        with gr.Column(scale=1, min_width=300):
            with gr.Column(elem_classes="upload-section"):
                gr.Markdown("### Upload Image")
                image_input = gr.Image(label="Upload Food Image", type="pil")
                submit_btn = gr.Button("Submit", elem_classes="btn-primary")
            # Allergen Impact card placed under the image preview
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

        # Right column: Two separate boxes for Nutrient Information and Ingredients
        with gr.Column(scale=1, min_width=400):
            # Nutrient Information Card
            with gr.Column(elem_classes="card"):
                gr.Markdown("### Nutrient Information")
                nutrient_message = gr.Markdown("")
                nutrient_table = gr.Dataframe(
                    value=default_df,
                    interactive=True,
                    label="Nutrient Information"
                )
                save_btn = gr.Button("Save Nutrient Table", elem_classes="btn-primary")
            # Ingredients Card
            with gr.Column(elem_classes="card"):
                gr.Markdown("### Ingredients")
                ingredients_text_output = gr.Markdown(value="No ingredients extracted yet.")

    # Note: Remove stream=True from here.
    submit_btn.click(
        fn=process_image,
        inputs=[image_input],
        outputs=[nutrient_table, ingredients_text_output, impact_text_output, allergen_state, nutrient_message]
    )

    # "Click More" button to reveal allergen details.
    click_more.click(
        fn=show_allergens,
        inputs=[allergen_state],
        outputs=[allergen_table]
    )

    # Save button callback
    save_btn.click(
        fn=save_table,
        inputs=[nutrient_table],
        outputs=[]
    )

# Enable the queue to support generator (streaming) outputs.
demo.queue()
demo.launch(share=False)
