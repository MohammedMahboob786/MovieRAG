# MovieRAG

MovieRAG is an application that takes an audio file as input and identifies the movie or TV series it belongs to. The application utilizes advanced Natural Language Processing (NLP) techniques and machine learning models to achieve accurate results. 

## Features

- **Audio-to-Text Conversion**: Uses the Whisper model to convert audio files into text.
- **Data Preprocessing**: Cleans and processes raw data using regex and BeautifulSoup to extract only useful content and remove HTML tags.
- **Text Chunking and Embedding**: Splits text data into chunks using RecursiveCharacterTextSplitter and generates embeddings using the "thenlper/gte-small" BERT model.
- **Similarity Search**: Utilizes Chroma Vector Database for efficient similarity search to find the matching movie name.
- **Scalable Indexing**: Processes 12,380 entries from a larger dataset, generating a total of 800,000 text chunks for robust movie identification.

## Workflow

1. **Data Preprocessing**: 
   - Extracts useful data using regex.
   - Removes any HTML tags with BeautifulSoup.
   
2. **Indexing**:
   - Uses 15% of original movie/series data (12,380 entries from a total of 82,498).
   - Loads data using Langchain's CSVLoader and splits into chunks of 500 characters with an overlap of 50.
   - Generates embeddings with the "thenlper/gte-small" BERT model and stores them in Chroma Vector Database.

3. **Application**:
   - Converts input audio to text using the Whisper model.
   - Transforms the text into embeddings.
   - Searches for similar embeddings in the vector database to retrieve the relevant chunk.
   - Passes the chunks to LLM to get the movie name.

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: For creating web application.
- **Whisper Model**: For audio-to-text conversion.
- **Langchain**: Utilized for data loading, chunking, ChromaDB initiallization, storing, retrieving, llm, chaining.
- **ChromaDB**: Vector database for storing and searching embeddings.
- **Regex & BeautifulSoup**: For data cleaning and preprocessing.
- **Hugging Face**: BERT model ("thenlper/gte-small") for generating text embeddings.

 
