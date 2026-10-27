# Jarvis — LLM Chatbot

A Python-based conversational chatbot powered by an LLM through the Groq API.

## Features

* Conversational interaction with an LLM
* Streaming responses
* Conversation history management
* Periodic conversation summarization
* Approximate input token counting
* Token usage tracking when supported by the API
* Basic error handling
* Environment-based API key configuration

## Tech Stack

* Python
* Groq API
* OpenAI Python SDK
* tiktoken
* python-dotenv

## Requirements

* Python 3.10 or newer
* A Groq API key
* Internet connection

## Installation

Clone the repository:


git clone https://github.com/DaniyalBadrlou/Jarvis-LLM-Chatbot.git
cd Jarvis-LLM-Chatbot


Create a virtual environment:


python -m venv .venv


Activate it on Windows:


.venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


## Configuration

Create a `.env` file in the project root:

GROQ_API_KEY=your_groq_api_key_here


Replace the placeholder with your own API key.

Never commit your actual API key to GitHub.

## Usage

Run the chatbot:

python chatbot.py


Type `q` to exit the conversation.

## Notes

* The configured model name must be available through the Groq API.
* Token counts from `tiktoken` are approximate for models that use a different tokenizer.
* Token usage and estimated cost depend on API support and the pricing configuration.
* Conversation summarization is a basic implementation and may require further testing for long conversations.

## Future Improvements

* Improved conversation memory
* More robust context-window management
* Unit tests
* Configuration for different LLM providers
* Better token usage and cost tracking

## Author

DaniyalBadrlou
