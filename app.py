# import streamlit as st
# import pandas as pd
# st.title("A Simple Streamlit Web App")
# name = st.text_input("Enter your name", "")
# st.write(f"Hello {name}!")
# x = st.slider("Select an integer x", 0, 10, 1)
# y = st.slider("Select an integer y", 0, 10, 1)
# df = pd.DataFrame({"x": [x], "y": [y] , "x + y": [x + y]}, index = ["addition row"])
# st.write(df)
#######################################
# from flask import Flask
# import streamlit as st
# m=""
# app = Flask(__name__)

# @app.route('/')
# def index():
#      m="ffffffffffffffffff"
# @app.route('/input')
# def index1():
#      m="ffffffffff22222ffffffff"


###########################################
import streamlit as st


import json

def create_json_object(input_data,response):
    data = {
        'input': input_data,
        'response': response,
        'accuracy': 0.85
    }
    data = {
        'response': response,
    }
    return json.dumps(data)


# Get the query parameters from the URL
query_params = st.experimental_get_query_params()

# Get the value of the 'input' parameter
input_value = query_params.get('input', [''])[0]

import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

@st.cache(allow_output_mutation=True)
def initialize_data():
    # Load the T5 model and tokenizer
    model_name = 'malmarjeh/t5-arabic-text-summarization'#'t5-base'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Perform any other necessary initialization or computations
    # ...

    return model, tokenizer

# Run the initialization function
model, tokenizer = initialize_data()

# # Streamlit app code
# st.write("T5 Model Initialized!")

# # Get user input
# input_text = st.text_input("Enter text to summarize:")

# # Check if input_text is not empty
# if input_text:
#     # Tokenize the input text
#     inputs = tokenizer.encode(input_text, return_tensors='pt')

#     # Generate the summary using the T5 model
#     summary_ids = model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)
#     summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     # Display the summary
#     st.write("Summary:")
#     st.write(summary_text)
import requests
from bs4 import BeautifulSoup

from urllib.parse import unquote




def get_arabic_wikipedia_link(query):
    # Send a GET request to the Google search results page
    response = requests.get(f"https://www.google.com/search?q={query}&hl=ar")

    # Parse the HTML content using BeautifulSoup
    #soup = BeautifulSoup(response.content, "html.parser")
    soup = BeautifulSoup(response.content, "html.parser", from_encoding="utf-8")

    # Find the first Wikipedia link that contains the title "Elon Musk"
    wikipedia_link = None
    for link in soup.find_all("a"):
        href = link.get("href")
        print(f"href {href}")
        if href.startswith("/url?q=https://en.wikipedia.org/wiki/") or "/url?q=https://ar.wikipedia.org/wiki/" in href :
            wikipedia_link = href.split("/url?q=")[1]
            wikipedia_link = wikipedia_link.split("&")[0]
            #print(f"wikipedia_link {wikipedia_link}")

            break

    # Print the URL of the Wikipedia page on Elon Musk
    if wikipedia_link is not None:
        print("im here before  unquote")
        print(wikipedia_link)
        decoded_url = unquote(wikipedia_link)
        decoded_url = unquote(decoded_url)
        print("im here after  unquote")
        print(decoded_url)
        #print(unicode_string)
        return decoded_url


    else:
        print("Wikipedia page not found.")
        return "Wikipedia page not found."



def scrape_wikipedia_sections(url):
    # Send a GET request to the URL and get the HTML content
    response = requests.get(url)
    html_content = response.text

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all the section headings and their contents
    sections = soup.select(".mw-headline")
    for section in sections:
        section_title = section.get_text()
        section_content = section.find_next("p")
        if section_content is not None:
            section_content = section_content.get_text()
            print(f"{section_title}\n{section_content}\n")
            print(f"===================summarize_sections=======================")
            summarization = summarize_sections(str(section_content))
            print(summarization)
            print(f"===================end=======================")
            print(f"===================summarize_sections1=======================")
            summarization1 = summarize_sections1(str(section_content))
            print(summarization1)
            print(f"===================end=======================")
    return summarization


def summarize_sections1(input_text):

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=512, early_stopping=True, length_penalty=2.0)
    final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(final_summary)
    return final_summary


def summarize_sections(input_text):

    sentences = input_text.split(".")
    input_text = ".".join(sentences[:6]) + "."
    num_sentences = min(6, len(sentences))

    # Initialize the final summary
    final_summary = ""
    # Concatenate the first 3 sentences that end with a period
    for i in range(num_sentences):
        sentence = sentences[i].strip()
        if sentence:
            #final_summary += sentence + ". "
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
            summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=512, early_stopping=True, length_penalty=2.0)
            final_summary += tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return final_summary


if input_value:
    # Get the URL of the Arabic Wikipedia page on Elon Musk
    wikipedia_url = get_arabic_wikipedia_link(input_value)

    # Scrape the sections and their contents from the Arabic Wikipedia page
    if wikipedia_url is not None:
         Summary=   scrape_wikipedia_sections(wikipedia_url)
    else:
        print("Arabic Wikipedia page not found.")
        Summary= "nooooooo"
#     # Tokenize the input text
#     inputs = tokenizer.encode(input_value, return_tensors='pt')

#     # Generate the summary using the T5 model
#     summary_ids = model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)
#     summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    json_object = create_json_object(wikipedia_url,Summary)
    st.json(json.loads(json_object))  # Parse JSON object for display

    # Display the summary
#     st.write("Summary:"+Summary)
#     st.write(wikipedia_url)
# # Use the input value
# def page_home():
#     st.title("Home")
#     st.write("Input value:", input_value)
#     # Add content specific to the Home page

# def page_about():
#     st.title("About")
#     st.write("This is the About page.")
#     # Add content specific to the About page

# def page_contact():
#     st.title("Contact")
#     st.write("You can contact us here.")
#     # Add content specific to the Contact page

# # Sidebar navigation
# sidebar_options = ["Home", "About", "Contact"]
# selected_option = st.sidebar.selectbox("Navigation", sidebar_options)

# # Conditionally render content based on selected option
# if selected_option == "Home":
#     page_home()
# elif selected_option == "About":
#     page_about()
# elif selected_option == "Contact":
#     page_contact()


