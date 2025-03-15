import boto3
import os

os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
os.environ["AWS_DEFAULT_REGION"] = aws_region

# Initialize Textract client
client = boto3.client(
    "rekognition",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ["AWS_DEFAULT_REGION"]
)

def extract_text_with_rekognition(image_bytes):
    try:
        response = client.detect_text(Image={'Bytes': image_bytes})
        # Extract text
        detected_text = [text['DetectedText'] for text in response.get('TextDetections', [])]
        detected_text = ' '.join(detected_text)
        print("✅ Amazon Rekognition Extraction Successful!\n")
        #return ' '.join(detected_text)
        return detected_text
    except Exception as e:
        print("❌ Amazon Rekognition failed:", str(e))
        print("🔄 Falling back to Tesseract OCR.")
        return None

if __name__ == '__main__':
    image_path = "C:/Users/rampa/Downloads/northwestern/northwestern/capstone_project/image_extraction/downloaded_images_2/image_3553366.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    text = extract_text_with_rekognition(image_bytes)
    print("success")



