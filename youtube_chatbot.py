

import os

"""# Install Libraries"""
from dotenv import load_dotenv
load_dotenv()


from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings


import youtube_transcript_api
print(dir(youtube_transcript_api.YouTubeTranscriptApi))

"""Step 1a - Indexing (Document Ingestion)"""

video_id = "Gfr50f6ZBvo"  # only the id not thee full URL

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

len(chunks)

"""Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Space)"""

embedder = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(chunks, embedder)

#vector_store.index_to_docstore_id

#vector_store.get_by_ids(["900318bd-d0c0-43dd-af81-dd6563314a03"])

"""Step 2 Retrieval"""

retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs={"k":4})

#retriever

#retriever.invoke("what is deep mind?")[0].page_content

"""Step 3 Augmentation"""

llm = ChatGroq(
    model_name = "llama-3.3-70b-versatile",
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

question = "is the topic of alliens discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({'context':context_text, 'question':question})

"""#Step 4 - Generation"""

answer = llm.invoke(final_prompt)
#print(answer.content)

"""#Building Chain"""

from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context':retriever|RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

#parallel_chain.invoke("who is Demis")

parser = StrOutputParser()

main_chain = parallel_chain| prompt | llm | parser

answer = main_chain.invoke("Can you summarize the video")

print(answer)