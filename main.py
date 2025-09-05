import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from PIL import Image, ImageOps
import fitz  # PyMuPDF
import pytesseract
import io
import json

# --- Helper Functions ---

def pdf_to_images(pdf_bytes):
    """Converts a PDF file in bytes to a list of PIL Images."""
    images = []
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        # --- IMPORTANT CHANGE FOR DEPLOYMENT ---
        # Using dpi=150 instead of 300 to reduce memory usage on cloud platforms.
        # This is the key to preventing memory-related crashes on Streamlit Cloud.
        dpi_setting = 150
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi_setting)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            images.append(image)
        pdf_document.close()
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []
    return images

def preprocess_image(image, options):
    """Applies selected pre-processing steps to a PIL image."""
    processed_image = image
    if 'grayscale' in options:
        processed_image = ImageOps.grayscale(processed_image)
    if 'invert' in options:
        # Invert works best on grayscale images
        if processed_image.mode != 'L':
            processed_image = ImageOps.grayscale(processed_image)
        processed_image = ImageOps.invert(processed_image)
    return processed_image


def perform_ocr(image, bounding_box, language, preprocessing_options):
    """Performs OCR on a cropped area of an image with pre-processing."""
    try:
        left, top, width, height = (
            int(bounding_box['left']),
            int(bounding_box['top']),
            int(bounding_box['width']),
            int(bounding_box['height'])
        )
        if width > 0 and height > 0:
            cropped_image = image.crop((left, top, left + width, top + height))
            
            # Apply selected pre-processing to the cropped image
            processed_crop = preprocess_image(cropped_image, preprocessing_options)

            lang_code = 'eng'
            if language == 'Hindi':
                lang_code = 'hin'
            elif language == 'English + Hindi':
                lang_code = 'eng+hin'
            
            text = pytesseract.image_to_string(processed_crop, lang=lang_code)
            return text.strip().replace('\n', ' ') # Clean up newlines
    except Exception as e:
        st.warning(f"Could not perform OCR on a region: {e}")
    return ""

# --- Streamlit App ---

st.set_page_config(layout="wide")

st.title("📄 AI-Powered Document Data Extractor")
st.markdown("Upload a PDF, define data fields (or load a template), and extract structured data from all pages.")

# --- Session State Initialization ---
if 'pdf_images' not in st.session_state:
    st.session_state.pdf_images = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'field_names' not in st.session_state:
    st.session_state.field_names = {}
if 'canvas_json' not in st.session_state:
    st.session_state.canvas_json = None


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    # 1. File Upload
    uploaded_file = st.file_uploader("1. Upload a PDF file", type="pdf")

    if uploaded_file and st.button("Clear and Restart"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if uploaded_file and not st.session_state.pdf_images:
        with st.spinner("Processing PDF... (This may take a moment)"):
            st.session_state.pdf_images = pdf_to_images(uploaded_file.getvalue())
        if st.session_state.pdf_images:
            st.success(f"PDF processed: {len(st.session_state.pdf_images)} pages.")
        else:
            st.error("Could not process the PDF. It might be corrupted or too large.")

    if st.session_state.pdf_images:
        st.header("🛠️ Extraction Settings")
        # 2. Language Selection
        language = st.selectbox("2. Select OCR Language", ('English', 'Hindi', 'English + Hindi'))
        st.session_state.language = language

        # 3. Image Pre-processing
        st.session_state.preprocessing_options = st.multiselect(
            "3. Image Pre-processing (for better OCR)",
            ['grayscale', 'invert'],
            help="Grayscale can improve contrast. Invert can help with light text on dark backgrounds."
        )

        # 4. Template Management
        st.header("💾 Template Management")
        
        # Load Template
        uploaded_template = st.file_uploader("Load Template (.json)", type="json")
        if uploaded_template:
            template_data = json.load(uploaded_template)
            st.session_state.canvas_json = template_data.get("regions", None)
            st.session_state.field_names = template_data.get("field_names", {})
            st.success("Template loaded!")
            st.rerun()


# --- Main App Body ---
if st.session_state.pdf_images:
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Step 1: Draw Regions on a Template Page")
        
        page_to_preview = st.slider(
            "Select page to use as template", 1, len(st.session_state.pdf_images), 1
        ) - 1

        bg_image = st.session_state.pdf_images[page_to_preview]
        canvas_width = 800
        canvas_height = canvas_width * (bg_image.height / bg_image.width)

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            background_image=bg_image,
            update_streamlit=True,
            height=int(canvas_height),
            width=int(canvas_width),
            drawing_mode="rect",
            initial_drawing=st.session_state.canvas_json,
            key="canvas",
        )
        if canvas_result.json_data:
            st.session_state.canvas_json = canvas_result.json_data


    with col2:
        if st.session_state.canvas_json and st.session_state.canvas_json.get("objects"):
            st.subheader("Step 2: Name Your Data Fields")
            regions = st.session_state.canvas_json["objects"]
            
            with st.form(key="field_names_form"):
                for i, region in enumerate(regions):
                    region_key = f"region_{i}_{int(region['left'])}_{int(region['top'])}"
                    default_name = st.session_state.field_names.get(region_key, f"Field_{i+1}")
                    st.session_state.field_names[region_key] = st.text_input(
                        f"Name for Region {i+1}", value=default_name, key=f"field_name_{i}"
                    )
                if st.form_submit_button("Save Field Names"):
                    st.success("Field names saved!")

            # Template Download
            template_to_save = {
                "regions": st.session_state.canvas_json,
                "field_names": st.session_state.field_names
            }
            st.download_button(
                label="📥 Save as Template",
                data=json.dumps(template_to_save, indent=2),
                file_name="extraction_template.json",
                mime="application/json",
            )

            st.subheader("Step 3: Extract Data")
            if st.button("🚀 Extract Data from All Pages", type="primary"):
                original_w, original_h = bg_image.size
                scale_w = original_w / canvas_width
                scale_h = original_h / canvas_height
                
                scaled_boxes, field_names_list = [], []

                for i, region in enumerate(regions):
                    region_key = f"region_{i}_{int(region['left'])}_{int(region['top'])}"
                    field_names_list.append(st.session_state.field_names.get(region_key, f"Field_{i+1}"))
                    scaled_boxes.append({
                        "left": region["left"]*scale_w, "top": region["top"]*scale_h,
                        "width": region["width"]*scale_w, "height": region["height"]*scale_h
                    })

                with st.spinner("Extracting data from all pages... This can take a while."):
                    all_pages_data = []
                    for i, page_image in enumerate(st.session_state.pdf_images):
                        page_data = {"Page": i + 1}
                        for j, box in enumerate(scaled_boxes):
                            field_name = field_names_list[j]
                            page_data[field_name] = perform_ocr(page_image, box, st.session_state.language, st.session_state.preprocessing_options)
                        all_pages_data.append(page_data)
                    st.session_state.extracted_data = pd.DataFrame(all_pages_data)
                
                st.success("Data extraction complete!")

# --- Data Preview and Download Section ---
if st.session_state.extracted_data is not None:
    st.header("📊 Extracted Data Preview")
    st.info("Review the extracted data below. You can make edits directly in the table before downloading.")
    
    edited_df = st.data_editor(st.session_state.extracted_data, num_rows="dynamic", use_container_width=True)
    
    csv = edited_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Edited Data as CSV",
        data=csv,
        file_name='extracted_multi_region_data.csv',
        mime='text/csv',
        type="primary"
    )

if not uploaded_file:
    st.info("Welcome! Please upload a PDF using the sidebar to begin.")

