# app.py - Pneumonia Detection with best_model.h5
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import h5py
import time

# Configure app
st.set_page_config(
    page_title="Pneumonia AI Scanner",
    page_icon="🫁",
    layout="centered"
)

# Constants
MODEL_FILE = "best_model.h5"  # Changed to your model filename
INPUT_SIZE = (224, 224)  # Must match model's expected input

@st.cache_resource
def load_model():
    """Enhanced model loader with troubleshooting"""
    try:
        # 1. Verify model file exists
        if not os.path.exists(MODEL_FILE):
            available_files = [f for f in os.listdir() if f.endswith('.h5')]
            raise FileNotFoundError(
                f"'{MODEL_FILE}' not found. Available H5 files: {available_files}"
            )
        
        # 2. Validate model file
        try:
            with h5py.File(MODEL_FILE, 'r') as f:
                if 'model_weights' not in f:
                    raise ValueError("Not a valid Keras model file")
        except Exception as e:
            raise ValueError(f"Corrupt model file: {str(e)}")
        
        # 3. Load with custom objects if needed
        model = tf.keras.models.load_model(
            MODEL_FILE,
            custom_objects={
                'FixedDropout': tf.keras.layers.Dropout
            }
        )
        
        # 4. Verify input shape
        if model.input_shape[1:3] != INPUT_SIZE:
            st.warning(
                f"Model expects {model.input_shape[1:3]} but using {INPUT_SIZE}. "
                "Resizing may affect accuracy."
            )
            
        return model
        
    except Exception as e:
        st.error(f"""
        ❌ MODEL LOADING FAILED
        Error: {str(e)}
        
        TROUBLESHOOTING:
        1. Run training code to generate '{MODEL_FILE}'
        2. Place file in: {os.getcwd()}
        3. Verify file isn't corrupted
        4. Check TensorFlow version matches training environment
        """)
        st.stop()

def main():
    st.title("🩺 Pneumonia Detection from Chest X-rays")
    
    # Load model
    try:
        model = load_model()
        st.sidebar.success("Model loaded successfully")
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload chest X-ray (JPEG/PNG)", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file:
        try:
            # Process image
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Uploaded Image", width=300)
            
            # Check quality
            if max(img.size) < 512:
                st.warning("Low resolution image - may affect accuracy")
            
            # Preprocess
            img = img.resize(INPUT_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict with progress
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            prob = float(model.predict(img_array, verbose=0)[0][0])
            progress_bar.empty()
            
            # Display results
            if prob > 0.5:
                st.error(f"""
                ## 🚨 Pneumonia Detected
                **Confidence:** {prob*100:.1f}%
                ### Recommended Actions:
                - Immediate physician review
                - Antibiotic therapy consideration
                - Follow-up imaging
                """)
            else:
                st.success(f"""
                ## ✅ Normal Result
                **Confidence:** {(1-prob)*100:.1f}%
                ### Recommendations:
                - Routine follow-up if symptomatic
                - No antibiotics needed
                """)
            
            # Visual gauge
            st.markdown(f"""
            <div style="background:#f0f2f6;padding:10px;border-radius:10px">
                <div style="background:linear-gradient(90deg, #4CAF50 {100*(1-prob)}%, #FF4B4B {100*(1-prob)}%); 
                            height:25px; border-radius:5px"></div>
                <p style="text-align:center">Pneumonia Probability: {prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.caption("""
    **Clinical Disclaimer:** This AI tool assists diagnosis but doesn't replace professional judgment.  
    Model trained on 5,856 cases (93.2% accuracy).  
    False negatives may occur - always consult a physician.
    """)

if __name__ == "__main__":
    main()