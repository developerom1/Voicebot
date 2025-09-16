from dotenv import load_dotenv
load_dotenv()
import os
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Test API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("API key not found")
    exit(1)

print("API key loaded successfully")

# Test LLM initialization
llm = OpenAI(api_key=api_key, temperature=0.7)
print("LLM initialized")

# Test memory
memory = ConversationBufferMemory()
print("Memory initialized")

# Test chain
chain = ConversationChain(llm=llm, memory=memory, verbose=False)
chain.memory.chat_memory.add_ai_message(
    "You are a helpful and friendly customer service agent for an ecommerce store. "
    "You can assist with product inquiries, order status, returns, shipping, and general customer support. "
    "Be polite, informative, and efficient in your responses."
)
print("Chain initialized")

# Test prediction
test_input = "What is the status of my order?"
response = chain.predict(input=test_input)
print(f"Test input: {test_input}")
print(f"Agent response: {response}")

print("LangChain agent test passed")

# Test TTS
import pyttsx3
engine = pyttsx3.init()
print("TTS engine initialized")

# Test speech recognition (without mic)
import speech_recognition as sr
recognizer = sr.Recognizer()
print("Speech recognizer initialized")
