import streamlit as st
from langchain_openai import ChatOpenAI
from PIL import Image
import io
import base64
import os

# -----------------
# CONFIG
# -----------------
st.set_page_config(page_title="Image Feature Extractor", layout="centered")

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # Add this line before instantiating embeddings

# -----------------
# LLM SETUP
# -----------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# -----------------
# PROMPT
# -----------------
SYSTEM_PROMPT = """
You are an expert in image recognition and visual analysis.
Given an image, identify ALL notable features, objects, and attributes.
Return the output as a bullet list, covering:

- Main objects
- Brands or logos
- Category (e.g., fast food, clothing, electronics)
- Related or implied items
- Context or environment clues

Be comprehensive but concise.

Analyze the given image and output a JSON object with these fields:
        1. image_description (string)
        2. detected_offer_category (string) from: grocery, gas, auto, home, pet, food, health, cosmetics,
           entertainment, shopping, travel, other, home_decor, home_improvement, vacation,
           subscription_Services, apparels, shoes
        3. detected_offer_location (string) from: in-store, on-line, pay at pump, single, none
        4. detected_purchase_type (string) from: subscription, membership, fuel, none
        5. detected_merchant_name (string) - possible merchant(s) in image
        6. remaining_days_count (integer) - estimated or null if unknown
        7. any_other_detected_features (array of strings)

        Return ONLY valid JSON, no explanation.
        Image: {image_url}
"""

# -----------------
# STREAMLIT UI
# -----------------
st.title("🔍 Image Feature Extractor (GPT-4o)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Image"):
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Send to GPT-4o with proper image format
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please analyze this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"
                            }
                        }
                    ]
                }
            ]
        )

        st.subheader("Detected Features:")
        st.write(response.content)

