# AI chatbot
In this project, AI chatbot app was created.


App allows user to choose from two modes - without documents and with documents uploaded in various formats - and have a conversation with AI chatbot which could answer to the user's questions taking into account provided documents and past questions within a session context. 

# User can:

- Upload documents (optional);
- Choose OpenAI model;
- Choose LLM's temperature (model's creativity);
- Choose LLM's number of tokens (model's response length);
- See conversation history within session;
- Download a history of the conversation session;
- Exit conversation session and start all over again.

# Key features of AI chatbot
- Once user attempts to jailbreak via input fields, AI chatbot prevents it by warning user and requesting to change the input;
- AI chatbot avoids engaging in harmful, unethical, or biased discussions and kindly inform user about this;
- User can upload one or multiple documents which will be stored in temporary memory and AI chatbot will adjust responses accordingly;
- User can have a fluent and natural conversation with AI chatbot since it remembers previous questions and responses within a particular session; 
- Implemented using LangChain;
- Conversation could be saved in CSV format;
- Publically available in the Internet for convenient usage.

AI chatbot app has been build as a single-page website using Streamlit under this [link](https://aichatbot-9aa5zdmjfdrkxh9hehrdwv.streamlit.app/) .

# Instructions for Public Deployment
- Save files from folder Deployment in public github repository;
- Create account in Streamlit public;
- Get OpenAI API key from openai.com website;
- Connect app file and requirements file to Streamlit;
- In Streamlit secrecy settings put your API details;
- Deploy AI chatbot app.

# Instructions for Local Setup
* Prerequisites:
  - Python 3.8 or higher
  - pip
  - Streamlit
  - OpenAI API key
    
* Installation:
  * Clone the repository;
      
  * Create a virtual environment:
            ```python -m venv venv```
            ```source venv/bin/activate  # On Windows: venv\Scripts\activate```
            
  * Install required packages:
            ```pip install -r requirements.txt```
            
  * Set your environment variables:
    * Set your LLM API key and any other required configs:
            ```set OPENAI_API_KEY=your-api-key```
    * Alternatively, create a .env file in the deployment/ directory and include: OPENAI_API_KEY=your-api-key
  * Run the Streamlit app:
            ```streamlit run app.py```
