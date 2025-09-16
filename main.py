import speech_recognition as sr
import pyttsx3
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    print("Please set OPENAI_API_KEY in .env file")
    exit(1)

# Initialize LangChain components
llm = OpenAI(api_key=openai_api_key, temperature=0.7)
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory, verbose=False)

# Set system prompt for ecommerce customer service
chain.memory.chat_memory.add_ai_message(
    "You are a helpful and friendly customer service agent for an ecommerce store. "
    "You can assist with product inquiries, order status, returns, shipping, and general customer support. "
    "Be polite, informative, and efficient in your responses."
)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 180)  # Speed of speech

# Initialize speech recognizer
recognizer = sr.Recognizer()

def speak(text):
    """Convert text to speech"""
    engine.say(text)
    engine.runAndWait()

def listen():
    """Listen for user speech input"""
    with sr.Microphone() as source:
        print("Listening for your query...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that.")
            return None
        except sr.RequestError:
            print("Speech recognition service unavailable.")
            return None

def main():
    """Main voicebot loop"""
    speak("Hello! I'm your ecommerce customer service assistant. How can I help you today?")
    
    while True:
        user_input = listen()
        if user_input:
            if user_input.lower() in ['quit', 'exit', 'bye']:
                speak("Thank you for using our service. Goodbye!")
                break
            
            # Get response from LangChain agent
            response = chain.predict(input=user_input)
            print(f"Assistant: {response}")
            speak(response)
        else:
            speak("I didn't catch that. Could you please repeat?")

if __name__ == "__main__":
    main()
