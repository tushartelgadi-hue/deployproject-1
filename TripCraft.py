import os
from dotenv import load_dotenv
from google import genai
from transformers import AutoTokenizer, AutoModelForCausalLM
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

client = genai.Client(api_key=api_key)

travel_faq = {
    "hi": "Hello! Welcome to the Travel Booking Helpdesk.",
    "hello": "Hi! How can I help you with your travel plans?",
    "destinations": "We offer bookings to popular destinations across India and abroad.",
    "booking": "You can book your trip through our website or by contacting our travel agent.",
    "flights": "You can search and book domestic and international flights.",
    "hotels": "We offer hotel booking options for different budgets.",
    "cancellation": "Cancellation charges depend on the airline, hotel, and booking policy.",
    "refund": "Refunds are processed according to the booking provider's refund policy.",
    "payment": "We accept supported online payment methods, including cards and UPI.",
    "documents": "For domestic travel, carry a valid accepted ID. International travel generally requires a passport and may require a visa.",
    "check-in": "Flight check-in can usually be completed online or at the airport.",
    "luggage": "Baggage allowance depends on the airline and ticket type.",
    "contact": "Please contact our travel support team for booking assistance.",
    "exit": "Thank you for using our Travel Booking Helpdesk. Have a safe journey!"
}

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def travel_booking_chat():
    print("Welcome to the Travel Booking Chatbot! ")

    while True:
        user_input = input("You: ").lower()

        if user_input == "exit":
          print("Chatbot: Thank you for reaching out! Have a great day!")
          break


        if user_input in travel_faq:
            print(f"Chatbot: {travel_faq[user_input]}")
            continue
        elif user_input not in travel_faq:
            inputs = tokenizer(user_input, return_tensors="pt")
            reply_ids = model.generate(**inputs)
            response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
            print(f"Chatbot: {response}")
            continue
travel_booking_chat()