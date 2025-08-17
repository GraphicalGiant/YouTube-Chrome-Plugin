
import os
from dotenv import load_dotenv
load_dotenv()


from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
#from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
#from langchain.embeddings import HuggingFaceEmbeddings

from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoTokenizer, AutoModel
import torch
from langchain_core.embeddings import Embeddings

class MiniLMembeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]
    
    def embed_query(self, text):
        return self._embed(text)
    
    def _embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", trauncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        masked_embeddings = embeddings*attention_mask
        summed = masked_embeddings.sum(1)
        counts = attention_mask.sum(1)
        mean_pooled = summed/ counts
        return mean_pooled.squeeze().tolist()

# Load Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")


def generate_answer(video_id, question):
    try:
        fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript_list = fetched_transcript.to_raw_data()
        """ transcript_list will be a list of dictionaries which contains text, and timestamps as the keys
        we only want text right now to will take each line and join it with a space in between"""

        #print(transcript_list)
        # flatten into plain text

        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        #print(transcript)

    except TranscriptsDisabled:
        print("No caption avilable for this video.")

    """Step 1b - Indexing (Text Splitting)"""

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    """Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Space)"""

    embedder = MiniLMembeddings()

    vector_store = FAISS.from_documents(chunks, embedder)

    #vector_store.index_to_docstore_id

    """Step 2 Retrieval"""

    retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs={"k":4})

    """Step 3 Augmentation"""

    llm = ChatGroq(
    api_key=groq_api_key,
    model_name = "llama-3.1-8b-instant",
    temperature = 0.2
    )

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        if the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables = ["context", 'question']

    )

    def format_docs(retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text

    parallel_chain = RunnableParallel({
        'context':retriever|RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()

    main_chain = parallel_chain| prompt | llm | parser

    return main_chain.invoke(question)





    