from gtts import gTTS
import pygame
import tempfile
import os
import time

def play_alert(message):
    """Generate and play voice alert"""
    try:
        # Initialize pygame mixer
        pygame.mixer.init()
        
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
        print(f"Audio error: {e}")
    finally:
        # Clean up
        try:
            pygame.mixer.quit()
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass