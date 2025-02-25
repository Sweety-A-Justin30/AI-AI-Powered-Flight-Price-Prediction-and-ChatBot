import os
import streamlit as st
import pandas as pd
import pickle
from fpdf import FPDF
from datetime import date, timedelta
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #ffc107;
        text-align: center;
    }
    </style>
    <p class="title"> ✈ AI-Powered Flight Price Prediction </p>
    """,
    unsafe_allow_html=True
)
import streamlit as st
import base64

def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()

    bg_style = f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{encoded_string}") no-repeat center fixed;
        background-size: cover;
    }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

# Call the function with your image file
set_background("background.jpg")  # Replace with your image filename


# Load the trained model
model_path = "flight_price_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Select travel date (starting from tomorrow)
tomorrow = date.today() + timedelta(days=1)
selected_date = st.date_input("Select Travel Date", min_value=tomorrow)

# Feature mappings
airlines = {"Vistara": 1, "Air_India": 2, "Indigo": 3, "GO_FIRST": 4, "AirAsia": 5, "SpiceJet": 6}
source_cities = {"Delhi": 1, "Mumbai": 2, "Bangalore": 3, "Kolkata": 4, "Hyderabad": 5, "Chennai": 6}
ticket_classes = {"Economy": 1, "Business": 2}

# User inputs
selected_source = st.selectbox("Select Source City", list(source_cities.keys()), key="source_city")
available_destinations = {k: v for k, v in source_cities.items() if k != selected_source}
selected_destination = st.selectbox("Select Destination City", list(available_destinations.keys()), key="destination_city")
selected_class = st.selectbox("Select Ticket Class", list(ticket_classes.keys()), key="ticket_class")
selected_airline = st.selectbox("Select Airline", list(airlines.keys()), key="airline")

# Convert inputs
source_city = source_cities[selected_source]
destination_city = available_destinations[selected_destination]
ticket_class = ticket_classes[selected_class]
airline = airlines[selected_airline]
days_left = (selected_date - date.today()).days

# Predict Button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[airline, source_city, destination_city, ticket_class, days_left]],
                              columns=["airline", "source_city", "destination_city", "class", "days_left"])

    # Make prediction
    predicted_price = model.predict(input_data)[0]

    # Apply tiered percentage reductions
    if predicted_price > 20000 and predicted_price <= 30000:
        predicted_price *= 0.70  # Reduce by 30%
    elif predicted_price > 30000 and predicted_price <= 40000:
        predicted_price *= 0.60  # Reduce by 40%
    elif predicted_price > 40000 and predicted_price <= 50000:
        predicted_price *= 0.50  # Reduce by 50%
    elif predicted_price > 50000:
        predicted_price *= 0.40  # Reduce by 60%

    # Ensure price does not fall below ₹5,000
    predicted_price = max(5000, predicted_price)

    # Store prediction in session state
    st.session_state["ticket_details"] = {
        "date": selected_date.strftime("%Y-%m-%d"),
        "airline": selected_airline,
        "source": selected_source,
        "destination": selected_destination,
        "class": selected_class,
        "price": predicted_price
    }

    # Display predicted price
    st.success(f"Predicted Flight Price: ₹{predicted_price:.2f}")

# Show Ticket Summary **Only After Prediction**
if "ticket_details" in st.session_state:
    ticket = st.session_state["ticket_details"]
    
    st.subheader("🛫 Ticket Summary")
    st.write(f"**📅 Date:** {ticket['date']}")
    st.write(f"**✈️ Airline:** {ticket['airline']}")
    st.write(f"**🛫 Source:** {ticket['source']}")
    st.write(f"**🛬 Destination:** {ticket['destination']}")
    st.write(f"**💺 Class:** {ticket['class']}")
    st.write(f"**💰 Price:** ₹{ticket['price']:.2f}")

    # Confirm Booking **Button appears only after prediction**
    if st.button("Confirm Booking"):
        st.success(f"Your ticket from {ticket['source']} to {ticket['destination']} is booked on {ticket['airline']}. Happy journey! 🎉")

        # Generate PDF Ticket
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Flight Ticket Confirmation", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, f"Date: {ticket['date']}", ln=True)
        pdf.cell(200, 10, f"Airline: {ticket['airline']}", ln=True)
        pdf.cell(200, 10, f"Source: {ticket['source']}", ln=True)
        pdf.cell(200, 10, f"Destination: {ticket['destination']}", ln=True)
        pdf.cell(200, 10, f"Class: {ticket['class']}", ln=True)
        pdf.cell(200, 10, f"Price: INR {ticket['price']:.2f}", ln=True)

        pdf_filename = "flight_ticket.pdf"
        pdf.output(pdf_filename, 'F')

        # Provide Download Option
        with open(pdf_filename, "rb") as file:
            st.download_button("Download Ticket", file, file_name=pdf_filename, mime="application/pdf")

st.markdown(
    """
    <style>
    .chatbot-header {
        font-size: 28px;
        font-weight: bold;
        color: #00aaff;
        text-align: center;
        padding: 5px;
        text-shadow: 0px 0px 10px rgba(255, 204, 0, 0.8);
        border-bottom: 2px solid #ffcc00;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
    }

    .chatbot-icon {
        font-size: 32px;
        filter: drop-shadow(0px 0px 6px rgba(220, 53, 69, 0.6));
    }
    div.stTextInput, div.stSelectbox, div.stDateInput {
        background-color: rgba(255, 255, 255, 0.7) !important;
        border-radius: 10px !important;
        padding: 5px !important;
        backdrop-filter: blur(5px);
    }
    button[kind="primary"] {
        background-color: rgba(255, 165, 0, 0.8) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 15px !important;
    }
    </style>

    <div class="chatbot-header">
        <span class="chatbot-icon">💬</span> Flight Booking Chatbot
    </div>
    """,
    unsafe_allow_html=True
)

# Set up LLM (Groq API)
llm = ChatGroq(model_name="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY"))

# Define Memory for Conversation
memory = ConversationBufferMemory(memory_key="history")

# Define a Custom Prompt (Ensuring Flight-Related Queries)
flight_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
    You are a flight booking assistant. You must only answer questions related to flights.
    If the question is unrelated to flights, kindly inform the user that you can only assist with flight-related queries.

    Previous conversation:
    {history}

    User: {input}
    Assistant:
    """
)

# Set up Conversation Chain
conversation = ConversationChain(llm=llm, memory=memory, prompt=flight_prompt)

# User Input for Chatbot
user_query = st.text_input("Ask any flight-related question:", key="flight_query_input")
if user_query:
    response = conversation.run(user_query)
    st.write("✈️ **Bot:**", response)
