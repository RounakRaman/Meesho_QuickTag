import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from PIL import Image
import torch

# Set up a lighter CPU-friendly model and tokenizer
model_id = "distilbert-base-uncased"
model = DistilBertForSequenceClassification.from_pretrained(model_id)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_id)

# Define function for generating attributes
def generate_attributes(image_path, category, attribute, possible_values):
    # Trim image whitespace if necessary
    image = Image.open(image_path).convert("RGB")
    
    placeholder = "<|image_1|>\n"
    text_prompt = f"{placeholder}. Choose from the following list: {possible_values}. What is the {attribute} of the product in the image (category: {category})? Provide a one-word answer from the list."

    # Tokenize the prompt and process it
    inputs = tokenizer(text_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_index = torch.argmax(outputs.logits, dim=1).item()
    response = possible_values[predicted_index]  # Simplified response based on prediction

    return response

# Streamlit App Setup
st.title("Meesho QuickTag: Simplifying Product Cataloging for Small Sellers")
st.write("Upload your product images below, either one-by-one or by uploading a folder.")

# Image Upload Options
uploaded_files = st.file_uploader("Upload Product Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# Category-specific attribute mapping
entity_unit_map = {
    "Men Tshirts": {
        "color": ["white", "black", "blue", "green", "yellow", "red", "grey", "maroon", "pink", "purple", "brown", "orange", "cream", "navy blue", "multicolor", "default", "not_available"],
        "neck": ["round neck", "polo", "v-neck", "henley", "crew neck", "mock neck", "turtle neck", "scoop neck", "mandarin collar", "notch neck", "default", "not_available"],
        "pattern": ["solid", "striped", "printed", "checked", "colorblocked", "tie-dye", "abstract", "camouflage", "textured", "geometric", "floral", "animal print", "default", "not_available"],
        "print_or_pattern_type": ["typography", "graphic", "abstract", "floral", "geometric", "camouflage", "solid", "default", "not_available"],
        "sleeve_length": ["short sleeves", "long sleeves", "three-quarter sleeves", "sleeveless", "cap sleeves", "default", "not_available"],
    },
    "Sarees": {
        "blouse_pattern": ["same as saree", "solid", "printed", "embroidered", "woven design", "default", "not_available"],
        "border": ["no border", "solid", "zari", "woven design", "temple border", "embroidered", "lace", "gotta patti", "contrast", "default", "not_available"],
        "border_width": ["big border", "small border", "medium border", "no border", "default", "not_available"],
        "color": ["navy blue", "white", "pink", "green", "yellow", "red", "orange", "cream", "purple", "maroon", "golden", "silver", "multicolor", "default", "not_available"],
        "occasion": ["party", "wedding", "festive", "casual", "formal", "traditional", "daily", "default", "not_available"],
        "ornamentation": ["tassels", "latkans", "zari work", "sequin", "thread work", "stone work", "jacquard", "embroidered", "lace", "default", "not_available"],
        "pallu_details": ["zari woven", "same as saree", "woven design", "embroidered", "default", "not_available"],
        "pattern": ["zari woven", "printed", "solid", "woven design", "embroidered", "default", "not_available"],
        "print_or_pattern_type": ["ethnic motif", "checked", "applique", "floral", "solid", "peacock", "botanical", "elephant", "default", "not_available"],
        "transparency": ["yes", "no", "default", "not_available"],
    },
    "Kurtis": {
        "color": ["navy blue", "white", "pink", "purple", "orange", "green", "red", "grey", "maroon", "yellow", "black", "multicolor", "blue", "default", "not_available"],
        "fit_shape": ["straight", "a-line", "default", "not_available"],
        "length": ["knee length", "calf length", "ankle length", "floor length", "default", "not_available"],
        "occasion": ["party", "casual", "festive", "formal", "default", "not_available"],
        "ornamentation": ["embroidered", "mirror work", "sequins", "lace", "thread work", "default", "not_available"],
        "pattern": ["solid", "printed", "embroidered", "colorblocked", "floral", "abstract", "default", "not_available"],
        "print_or_pattern_type": ["solid", "floral", "geometric", "ethnic motif", "abstract", "default", "not_available"],
        "sleeve_length": ["short sleeves", "three-quarter sleeves", "long sleeves", "sleeveless", "cap sleeves", "default", "not_available"],
        "sleeve_styling": ["regular", "bell sleeves", "flared sleeves", "puff sleeves", "cuffed sleeves", "default", "not_available"],
    },
    "Women Tshirts": {
        "color": ["white", "pink", "yellow", "maroon", "black", "multicolor", "blue", "green", "default", "not_available"],
        "fit_shape": ["regular", "loose", "boxy", "cropped", "slim", "default", "not_available"],
        "length": ["regular", "long", "crop", "default", "not_available"],
        "pattern": ["solid", "printed", "striped", "colorblocked", "tie-dye", "graphic", "default", "not_available"],
        "print_or_pattern_type": ["funky print", "quirky", "solid", "graphic", "typography", "floral", "default", "not_available"],
        "sleeve_length": ["short sleeves", "long sleeves", "three-quarter sleeves", "sleeveless", "cap sleeves", "default", "not_available"],
        "sleeve_styling": ["regular sleeves", "cuffed sleeves", "roll-up sleeves", "puff sleeves", "default", "not_available"],
        "surface_styling": ["applique", "ruffles", "embroidery", "lace", "default", "not_available"],
    },
    "Women Tops & Tunics": {
        "color": ["navy blue", "white", "pink", "peach", "green", "red", "maroon", "yellow", "black", "multicolor", "blue", "default", "not_available"],
        "fit_shape": ["boxy", "fitted", "regular", "loose", "default", "not_available"],
        "length": ["regular", "crop", "longline", "default", "not_available"],
        "neck_collar": ["stylised", "high neck", "square neck", "v-neck", "sweetheart neck", "round neck", "boat neck", "cowl neck", "collar neck", "default", "not_available"],
        "occasion": ["casual", "party", "formal", "default", "not_available"],
        "pattern": ["solid", "printed", "checked", "floral", "striped", "abstract", "default", "not_available"],
        "print_or_pattern_type": ["quirky", "solid", "floral", "graphic", "abstract", "typography", "default", "not_available"],
        "sleeve_length": ["long sleeves", "short sleeves", "three-quarter sleeves", "sleeveless", "cap sleeves", "default", "not_available"],
        "sleeve_styling": ["puff sleeves", "regular sleeves", "cuffed sleeves", "flared sleeves", "default", "not_available"],
        "surface_styling": ["waist tie-ups", "ruffles", "lace inserts", "embroidery", "knitted", "default", "not_available"],
    }
}

# Process images and display attributes
if uploaded_files:
    category = st.selectbox("Select Product Category", list(entity_unit_map.keys()))
    
    st.write("Generated Tags (Attributes) for Uploaded Images:")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image - {uploaded_file.name}")
        
        attribute_tags = []
        attributes = entity_unit_map[category]
        
        tag_display = ""
        for attribute, possible_values in attributes.items():
            tag = generate_attributes(uploaded_file, category, attribute, possible_values)
            tag_display += f"{attribute}: #{tag}\n"
            attribute_tags.append(f"{attribute}: #{tag}")
        
        st.write("Tags with Labels for Easy Copying:")
        st.text_area("Copy Tags Here:", tag_display, height=500, key=uploaded_file.name)

st.write("Meesho QuickTag will generate attribute tags based on the product's category, helping to avoid cataloging errors and improve visibility on search and recommendation systems.")
st.write("Once you have reviewed the tags, copy and use them to improve your product catalog on Meesho!")
