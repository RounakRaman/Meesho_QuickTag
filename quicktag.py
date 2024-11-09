import streamlit as st
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import os

# Set up the model and processor
model_id = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation="eager")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=8)

# Define function for generating attributes
def generate_attributes(image_path, category, attribute, possible_values):
    # Trim image whitespace if necessary
    # Implemented as per previous code for whitespace trimming
    image = Image.open(image_path).convert("RGB")
    
    placeholder = "<|image_1|>\n"
    messages = [{"role": "user", "content": f"{placeholder}. Choose from the following list: {possible_values}. What is the {attribute} of the product in the image (category: {category})? Provide a one-word answer from the list."}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=1000, temperature=0.0, do_sample=False)
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response

# Streamlit App Setup
st.title("Meesho QuickTag: Simplifying Product Cataloging for Small Sellers")
st.write("""
    Upload your product images below, either one-by-one or by uploading a folder. 
    
""")

# Image Upload Options
uploaded_files = st.file_uploader("Upload Product Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# Process images and display attributes
if uploaded_files:
    category = st.selectbox("Select Product Category", ["Men Tshirts", "Sarees", "Kurtis", "Women Tshirts", "Women Tops & Tunics"])
    entity_unit_map = {
        "Men Tshirts": {
        "color": ["white", "default", "black", "multicolor", "unknown"],
        "neck": ["round", "unknown", "polo"],
        "pattern": ["solid", "printed", "unknown"],
        "print_or_pattern_type": ["solid", "unknown", "typography", "default"],
        "sleeve_length": ["short sleeves", "long sleeves", "unknown"],
    },
    "Sarees": {
        "blouse_pattern": ["same as saree", "default", "solid", "same as border", "unknown"],
        "border": ["default", "no border", "solid", "temple border", "woven design", "unknown", "zari"],
        "border_width": ["unknown", "big border", "small border", "no border"],
        "color": ["navy blue", "white", "pink", "green", "cream", "yellow", "default", "multicolor", "unknown"],
        "occasion": ["party", "wedding", "daily", "traditional", "unknown"],
        "ornamentation": ["tassels and latkans", "unknown", "default", "jacquard"],
        "pallu_details": ["zari woven", "same as saree", "default", "woven design", "unknown"],
        "pattern": ["zari woven", "printed", "default", "solid", "woven design", "unknown"],
        "print_or_pattern_type": ["ethnic motif", "checked", "default", "applique", "floral", "solid", "peacock", "unknown", "botanical", "elephant"],
        "transparency": ["unknown", "yes", "no"],
    },
    "Kurtis": {
        "color": ["navy blue", "white", "pink", "purple", "orange", "green", "red", "grey", "maroon", "yellow", "black", "multicolor", "blue", "unknown"],
        "fit_shape": ["straight", "unknown", "a-line"],
        "length": ["unknown", "knee length", "calf length"],
        "occasion": ["unknown", "party", "daily"],
        "ornamentation": ["default", "unknown", "net"],
        "pattern": ["solid", "unknown", "default"],
        "print_or_pattern_type": ["solid", "unknown", "default"],
        "sleeve_length": ["short sleeves", "three-quarter sleeves", "unknown", "sleeveless"],
        "sleeve_styling": ["regular", "unknown", "sleeveless"],
    },
    "Women Tshirts": {
        "color": ["white", "pink", "yellow", "maroon", "default", "black", "multicolor", "unknown"],
        "fit_shape": ["regular", "loose", "unknown", "boxy"],
        "length": ["regular", "long", "unknown", "crop"],
        "pattern": ["solid", "printed", "unknown", "default"],
        "print_or_pattern_type": ["funky print", "quirky", "default", "solid", "graphic", "unknown", "typography"],
        "sleeve_length": ["short sleeves", "long sleeves", "unknown", "default"],
        "sleeve_styling": ["regular sleeves", "cuffed sleeves", "unknown"],
        "surface_styling": ["unknown", "default", "applique"],
    },
    "Women Tops & Tunics": {
        "color": ["navy blue", "white", "pink", "peach", "green", "red", "maroon", "yellow", "default", "black", "multicolor", "blue", "unknown"],
        "fit_shape": ["boxy", "default", "fitted", "regular", "unknown"],
        "length": ["regular", "crop", "unknown"],
        "neck_collar": ["stylised", "high", "square neck", "default", "v-neck", "sweetheart neck", "round neck", "unknown"],
        "occasion": ["unknown", "party", "casual"],
        "pattern": ["solid", "printed", "unknown", "default"],
        "print_or_pattern_type": ["quirky", "default", "solid", "floral", "graphic", "unknown", "typography"],
        "sleeve_length": ["long sleeves", "short sleeves", "three-quarter sleeves", "unknown", "sleeveless"],
        "sleeve_styling": ["puff sleeves", "default", "regular sleeves", "unknown", "sleeveless"],
        "surface_styling": ["waist tie-ups", "ruffles", "default", "applique", "unknown", "tie-ups", "knitted"],
    }
        # Add other categories with attributes...
    }
    
    st.write("Generated Tags (Attributes) for Uploaded Images:")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image - {uploaded_file.name}")
        
        attribute_tags = []
        attributes = entity_unit_map[category]  # Get attribute mappings for the selected category
        for attribute, possible_values in attributes.items():
            tag = generate_attributes(uploaded_file, category, attribute, possible_values)
            attribute_tags.append(f"#{tag}")
        
        # Display attributes with hashtags and copy option
        tags_str = " ".join(attribute_tags)
        st.write(f"Tags: {tags_str}")
        st.text_area("Copy these tags for your product", tags_str, key=uploaded_file.name)
st.write ("Meesho QuickTag will generate attribute tags based on the product's category,helping to avoid cataloging errors and improve visibility on search and recommendation systems.")
st.write("Once you have reviewed the tags, copy and use them to improve your product catalog on Meesho!")
