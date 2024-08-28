import streamlit as st
from openai import OpenAI
import pickle 
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Initialize OpenAI client, ChatOpenAI with API key
####################################################
f = open(".openai_secretkey.txt")
KEY = f.read().strip()
client = OpenAI(api_key = KEY)
chat_model = ChatOpenAI(openai_api_key=KEY)
####################################################


#Load the Embeddings model(thenlper/gte-small) saved in a pickle file
###########################################
file_path = 'embeddings model.pkl'

with open(file_path, 'rb') as file:
    embeddings_model = pickle.load(file)
###########################################



# Function to transcribe audio using OpenAI Whisper
##############################################################################
def transcribe_audio_and_fetch_relevant_movie(audio_file):
    try:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",  #performs multilingual speech recognition
            file=audio_file     #audio file by the user
        )
        subtitles = transcription.text
        movie =  RAG(subtitles) #Pass the subtitles to RAG function
        return movie
    
    except Exception as e:
        return str(e)
##############################################################################



def RAG(text):  
    # Initialize a ChromaDB Connection
    ##############################################################################################################
    CHROMA_PATH = "DataBase2"

    # Initialize the database connection
    try:
        db = Chroma(collection_name="vector_database",
                    embedding_function=embeddings_model,
                    persist_directory=CHROMA_PATH)
    except Exception as e:
        return f"Error initializing Chroma: {e}"
    ##############################################################################################################
    
    
       
    # Prompt Template
    ######################################################################################################################
    system_message = """
    You are an AI model with expertise in understanding and analyzing movie scripts
    and subtitles. Your task is to accurately identify the movie title based on
    a provided subtitle excerpt.

    -> Carefully analyze the provided subtitle text.
    -> Match it with the appropriate movie title by considering the context, characters, themes, and language used in the excerpt.
    -> Provide only the movie title as the final answer unless specifically requested for additional information."""

    USER_PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}
    Answer the question based on the above context: {question}.
    Provide a detailed answer.
    Don't justify your answers.
    Don't give information not mentioned in the CONTEXT INFORMATION.
    Do not say "according to the context" or "mentioned in the context" or similar.
    """
    SYSTEM_MESSAGE_PROMPT = SystemMessagePromptTemplate.from_template(system_message)
    
    prompt_template = ChatPromptTemplate.from_messages([SYSTEM_MESSAGE_PROMPT, USER_PROMPT_TEMPLATE])
    ######################################################################################################################

    
    
    # Output Parser
    ######################################################################################################################
    parser = StrOutputParser()
    ######################################################################################################################
    
    
    
    # Retriever
    ######################################################################################################################
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    ######################################################################################################################
    
    
    
    # Define a RAG Chain
    ######################################################################################################################
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    try:
        rag_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt_template | chat_model | parser
        query = text
        output = rag_chain.invoke(query)
    except Exception as e:
        return f"Error in RAG chain: {e}"
    ######################################################################################################################
    return output

# Streamlit interface
st.title("MovieRAG")

st.write("Upload an audio file, and we'll identify the movie it belongs to below.")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    if st.button("Get the Movie name"):
        with st.spinner("Processing..."):
            movie_name = transcribe_audio_and_fetch_relevant_movie(uploaded_file)
            st.success("Process completed!")
            st.chat_message("assistant").write(movie_name)
            
