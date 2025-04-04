import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit.components.v1 as components
import time
import os
import pygame
from gtts import gTTS
import tempfile

# Initialize pygame mixer once at app start
pygame.mixer.init()

def play_alert(message):
    """Generate and play voice alert"""
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
        
        # Generate speech with slower speed and clear pronunciation
        tts = gTTS(text=message, lang='en', slow=True)
        tts.save(temp_path)
        
        # Wait for file to be saved
        time.sleep(0.5)
        
        # Load and play
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        
        # Wait while audio is playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
    except Exception as e:
        st.error(f"Audio error: {e}")
    finally:
        # Clean up
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass

def detect_ppe(model, frame):
    results = model.predict(frame)
    output_frame = results[0].plot()
    
    # Get detected classes
    detected_classes = []
    for result in results:
        classes = result.boxes.cls.cpu().numpy()
        detected_classes = [model.names[int(cls)] for cls in classes]
    
    # Define required PPE items
    required_ppe = ["helmet", "vest", "gloves", "boots"]
    missing = [item for item in required_ppe if item not in detected_classes]
    
    return output_frame, missing

# --- App Initialization with Loading Animation ---
def initialize_app():
    """Shows loading animation while initializing app components"""
    loading_html = """
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;">
        <iframe src="https://lottie.host/embed/a5b6b121-853c-45c2-a4a7-80ef32fed6b3/J1y58IKznY.lottie" 
                width="300" height="300" frameborder="0" allowfullscreen>
        </iframe>
        <p style="font-size: 1.2rem; margin-top: 20px;">Loading PPE Inspector...</p>
    </div>
    """
    loading_placeholder = st.empty()
    loading_placeholder.markdown(loading_html, unsafe_allow_html=True)
    
    # Load heavy resources
    model = YOLO("models/best.pt")  # Replace with your actual model path
    
    # Simulate loading (remove time.sleep() in production)
    time.sleep(2) 
    
    # Clear loading animation
    loading_placeholder.empty()
    return model

# Initialize app and load model
model = initialize_app()

# --- Main App UI ---
st.title("🏗️ Construction PPE Inspector")
st.markdown("""
**Detects:** Helmets • Vests • Gloves • Safety Boots  
**Alerts:** Clear voice warnings for missing equipment
""")

# Add audio toggle in sidebar
with st.sidebar:
    st.markdown("### Settings")
    enable_audio = st.checkbox("Enable Voice Alerts", value=True)
    st.markdown("""
    ### About This App
    This tool helps construction site supervisors:
    - Automatically detect PPE compliance
    - Identify missing safety equipment
    - Improve worksite safety standards
    *Model accuracy may vary based on lighting and image quality*
    """)

# --- Image Upload Section ---
uploaded_file = st.file_uploader("Upload Site Photo", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    try:
        # Show processing animation
        with st.spinner('Analyzing PPE...'):
            processing_html = """
            <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; margin: 20px 0;">
                <iframe src="https://lottie.host/embed/a5b6b121-853c-45c2-a4a7-80ef32fed6b3/J1y58IKznY.lottie" 
                        width="200" height="200" frameborder="0" allowfullscreen>
                </iframe>
                <p style="font-size: 1rem; margin-top: 10px;">Processing image...</p>
            </div>
            """
            processing_placeholder = st.empty()
            processing_placeholder.markdown(processing_html, unsafe_allow_html=True)
            
            # Process image
            image = Image.open(uploaded_file).convert("RGB")
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output_frame, missing = detect_ppe(model, frame)
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            
            # Clear processing animation
            processing_placeholder.empty()
            
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(output_frame, caption="PPE Analysis")
        
        # Show missing equipment
        if missing:
            st.error(f"🚨 Missing PPE: {', '.join(missing)}")
            if enable_audio:
                play_alert(f"Warning! Missing safety equipment: {', '.join(missing)}")
        else:
            st.success("✅ All PPE properly equipped!")
            if enable_audio:
                play_alert("All safety equipment detected. Good job!")
            
    except Exception as e:
        st.error(f"Processing error: {str(e)}")

# --- Live Camera Section ---
if st.checkbox("Enable Live Inspection", help="Real-time PPE detection using webcam"):
    cam = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    stop_button = st.button("Stop Live Inspection")
    
    # Add cooldown for live alerts
    last_alert_time = 0
    alert_cooldown = 5  # seconds
    
    try:
        while cam.isOpened() and not stop_button:
            ret, frame = cam.read()
            if not ret:
                st.warning("Camera disconnected")
                break
                
            # Process frame with loading indicator
            with st.spinner('Live detecting...'):
                output_frame, missing = detect_ppe(model, frame)
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                
            # Display results
            frame_placeholder.image(output_frame, channels="RGB")
            
            # Update status with cooldown for alerts
            current_time = time.time()
            if missing:
                status_placeholder.error(f"⚠ Missing: {', '.join(missing)}")
                if enable_audio and (current_time - last_alert_time) > alert_cooldown:
                    play_alert(f"Warning! Missing: {', '.join(missing)}")
                    last_alert_time = current_time
            else:
                status_placeholder.success("All PPE detected")
                if enable_audio and (current_time - last_alert_time) > alert_cooldown:
                    play_alert("All equipment detected")
                    last_alert_time = current_time
                    
    finally:
        cam.release()
        st.info("Live inspection stopped")