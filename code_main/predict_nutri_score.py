import pickle
import pandas as pd
import numpy as np

model_path = "model/random_forest_model.pkl"

nutri_score_explanations = {
        'A': "🔹 **Excellent nutritional quality**\nLow in saturated fats, sugars, and salt. High in fiber and essential nutrients. A great choice for a balanced diet.",
        'B': "🔸 **Good nutritional quality**\nSlightly higher in fats or sugars but still a healthy option.",
        'C': "🟠 **Moderate nutritional quality**\nContains more processed ingredients, higher sugar, or fat content. Suitable in moderation.",
        'D': "🟥 **Poor nutritional quality**\nHigh in sugar, salt, and unhealthy fats. Limit consumption.",
        'E': "🚨 **Very poor nutritional quality**\nHighly processed, high in sugars, saturated fats, and salt. Consume sparingly.",
        'NA': "❓ **Not Available**\nNutri-score could not be determined."
    }

class NutriScorePredictor:
    def __init__(self):
        """
        Initializes the NutriScorePredictor by loading the model from a pickle file.

        Args:
            model_path (str): Path to the pickle file containing the trained model.
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Define the required features (order must match training data)
        self.features = [
            'energy_100g',
            'fat_100g',
            'saturated-fat_100g',
            'trans-fat_100g',
            'cholesterol_100g',
            'carbohydrates_100g',
            'fiber_100g',
            'proteins_100g',
            'salt_100g',
            'sodium_100g',
            'potassium_100g',
            'sugars_100g',
            'calcium_100g',
            'iron_100g',
            'fruits-vegetables-nuts-estimate-from-ingredients_100g'
        ]

        # Mapping from model output to Nutri-score labels
        self.nutri_score_mapping = {0: 'NA', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

    def predict(self, df: pd.DataFrame):
        """
        Predicts the Nutri-Score for the provided nutritional data.

        Args:
            df (pd.DataFrame): DataFrame containing two columns: 'nutrient' and 'per_100g'.
                Example:
                    nutrient                         per_100g
                    energy_100g                      31.00
                    fat_100g                         0.00
                    saturated_fat_100g               0.00
                    carbohydrates_100g               7.50
                    sugars_100g                      7.40
                    fiber_100g                       0.50
                    proteins_100g                    2.50
                    salt_100g                        0.07

        Returns:
            The predicted Nutri-score label if a single prediction is made, or a list of labels if multiple rows are provided.
        """
        # Create a dictionary mapping nutrient names to their corresponding values from the input DataFrame.
        nutrient_values = dict(zip(df['Nutrient'], df['per_100g']))

        # Define a mapping for converting input nutrient names to model's expected feature names.
        # (For example, the input uses underscores while the model expects a hyphen for "saturated-fat_100g".)
        name_mapping = {
            "saturated_fat_100g": "saturated-fat_100g",
            "trans_fat_100g": "trans-fat_100g"
            # Other nutrients are assumed to have matching names.
        }

        # Build the input row in the order required by the model.
        input_row = []
        for feature in self.features:
            # Determine the corresponding input key
            input_key = None
            # If this feature is mapped from a different naming convention, retrieve the original key.
            for key, mapped in name_mapping.items():
                if mapped == feature:
                    input_key = key
                    break
            if input_key is None:
                input_key = feature
            # Get the value for the nutrient; if not provided, default to 0.
            value = nutrient_values.get(input_key, 0)
            input_row.append(value)

        # Convert the input row into a numpy array with shape (1, number of features)
        input_data = np.array([input_row])

        # Generate predictions using the loaded model.
        predictions = self.model.predict(input_data)

        # Map numerical predictions to Nutri-score labels.
        mapped_predictions = [self.nutri_score_mapping.get(pred, 'Unknown') for pred in predictions]

        if len(mapped_predictions) == 1:
            return mapped_predictions[0]
        return mapped_predictions


if __name__ == '__main__':
    # Create a sample DataFrame with columns "nutrient" and "per_100g".
    sample_data = {
        'Nutrient': [
            'energy_100g',
            'fat_100g',
            'saturated_fat_100g',
            'carbohydrates_100g',
            'sugars_100g',
            'fiber_100g',
            'proteins_100g',
            'salt_100g'
        ],
        'per_100g': [31.00, 0.00, 0.00, 7.50, 7.40, 0.50, 2.50, 0.07]
    }

    sample_df = pd.DataFrame(sample_data)
    nutri_score_predictor = NutriScorePredictor()
    prediction,explanation = nutri_score_predictor.predict(sample_df)
    print("Predicted Nutri-Score:", prediction)
    print("Predicted Nutri-Score:", explanation)
