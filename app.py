import streamlit as st
import requests
import pandas as pd

# Streamlit app configuration
st.title("RAG-Enabled CRM System")
st.sidebar.header("Navigation")

# Sidebar options
options = st.sidebar.radio(
    "Choose an option",
    ["Upload Data", "Query with RAG", "Real-Time Call Dashboard", "Sentiment Analysis", "Real-Time Recommendation"]
)

API_URL = "http://127.0.0.1:8000/api"  # Backend API endpoint


if options == "Upload Data":
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload your text file", type=["txt", "csv"])

    if uploaded_file:
        if st.button("Upload"):
            # Read file contents
            if uploaded_file.name.endswith(".txt"):
                data = uploaded_file.read().decode("utf-8")
            elif uploaded_file.name.endswith(".csv"):
                data = "\n".join(pd.read_csv(uploaded_file)["text"].tolist())

            # Send data to the backend
            response = requests.post(f"{API_URL}/add-data", json={"data": data})
            if response.status_code == 200:
                st.success("Data uploaded successfully!")
            else:
                st.error("Failed to upload data.")

elif options == "Query with RAG":
    st.subheader("Ask Questions with RAG")
    query = st.text_area("Enter your query")

    if st.button("Get Answer"):
        # Send query to the backend
        response = requests.post(f"{API_URL}/recommendation", json={"query": query})
        if response.status_code == 200:
            response_data = response.json()
            # Display results
            st.write("### Query:")
            st.write(response_data["query"])
            st.write("### RAG Response:")
            st.write(response_data["recommendations"])
            st.write("### Retrieved Documents:")
            for doc in response_data.get("retrieved_documents", []):
                st.write(f"- {doc['metadata'].get('text', '')}")
        else:
            st.error("Failed to fetch the RAG response.")

elif options == "Real-Time Call Dashboard":
    st.subheader("Real-Time Call Dashboard")
    st.write("This section will display live call recommendations and prompts.")
    st.write("Coming Soon: Integrate live features here.")

elif options == "Sentiment Analysis":
    st.subheader("Real-Time Sentiment Analysis")
    text = st.text_area("Enter the text for sentiment analysis")

    if st.button("Analyze Sentiment"):
        # Send text to the backend for sentiment analysis
        response = requests.post(f"{API_URL}/sentiment", json={"text": text})
        if response.status_code == 200:
            response_data = response.json()
            st.write("### Input Text:")
            st.write(response_data["text"])
            st.write("### Sentiment Analysis Result:")
            st.write(response_data["sentiment_analysis"])
        else:
            st.error("Failed to analyze sentiment.")

elif options == "Real-Time Recommendation":
    st.subheader("Real-Time Recommendation")
    query = st.text_input("Enter your query for recommendation")

    if st.button("Get Recommendation"):
        # Send query to the backend
        response = requests.post(f"{API_URL}/recommendation", json={"query": query})
        if response.status_code == 200:
            response_data = response.json()
            st.write("### Query:")
            st.write(response_data["query"])
            st.write("### Recommendations:")
            st.write(response_data["recommendations"])
        else:
            st.error("Failed to fetch recommendations.")
