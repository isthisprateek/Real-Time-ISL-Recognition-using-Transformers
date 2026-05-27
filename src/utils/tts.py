"""
Text-to-Speech utility for ISL recognition.
"""
import pyttsx3
import threading
import logging

logger = logging.getLogger(__name__)

class TTSModule:
    def __init__(self, rate=150, volume=1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self.lock = threading.Lock()
        
    def speak(self, text):
        """Speak text in a separate thread to avoid blocking the main loop."""
        def _speak():
            with self.lock:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    logger.error(f"TTS Error: {e}")
        
        thread = threading.Thread(target=_speak)
        thread.start()

# Global instance
tts = TTSModule()
