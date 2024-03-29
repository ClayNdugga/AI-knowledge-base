# LLM-based Research Paper Abstract Generator

The project is avalible here: https://ai-knowledge-base.streamlit.app/

## Overview

This project demonstrates a serverless architecture for generating professional paper abstracts using **Large Language Models (LLMs)**. It employs a **Retrieval-Augmented Generation (RAG)** approach, integrating LLM prompt engineering with queries to a vector database for contextual relevance and accuracy. This solution showcases practices and tools that are crucial for modern, scalable applications. \
\
At the heart of this project lies the **RAG** approach, which significantly enhances the capability of LLMs by incorporating a retrieval step into the generation process. This methodology involves querying a vector database to find relevant information that contextually enriches the model's output. In the context of this application, when the user requests an abstract, RAG gets the 3 most relevant exisitng abstracts from the vector store, then sends them to the LLM with the intital prompt to increase the quality of the result.


## Application Across Domains

The principles and architecture demonstrated in this project can be translated to a wide range of applications beyond paper abstract generation. For instance, it can be adapted for:

- **Question Answering Systems**: Enhancing the accuracy and relevance of answers by retrieving similar questions or related information.
- **Content Recommendation**: Augmenting recommendation engines by incorporating retrieval-based context to personalize content.
- **Language Understanding Tasks**: Improving natural language processing applications by providing relevant background information for better context understanding.

## Project Architecture

<figure align="center">
  <img src="https://github.com/ClayNdugga/AI-knowledge-base/blob/main/assets/architecture.png?raw=true" alt="architecture"/>
  <figcaption style="text-align: center;">
    <i>Serverless Architecture</i>
  </figcaption>
</figure>

The diagram outlines the flow of data and interactions between the user, web application, and various AWS services, including the utilization of FAISS for vector search capabilities.

1. **Web Application**: A streamlit UI hosted on the public cloud serves as the front-end to take paper titles and findings
2. **Amazon API Gateway**: Manages and routes incoming API requests to the lambda function.
3. **AWS Lambda**: Handles incoming API request: \
   **3.1.** Process request to extract paper title and findings submitted by the user\
   **3.2.** Search S3 bucket using FAISS to find papers similiar to users query \
   **3.3.** The function creates a prompt, with the user's query and the similiar abstracts as context. It then asks the OPEN AI API to genrate a response. \
   **3.4.** The function returns the response to the API gateway, updating the user UI


4. **Vector Store Updates**: The S3 vector store is updated offline and uses FAISS. FAISS is a highly efficient library for similarity search and clustering of dense vectors that scales to billions of vectors. The raw text data containing paper titles and abstracts is stored in an offline DB.


### Technologies & Frameworks
![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![LangChain](https://img.shields.io/badge/-LangChain-blue?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/-Streamlit-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)


- **AWS**: For serverless compute, storage, and API management.
- **Docker**: To create Lambda layers that enable efficient and effective execution of the application's dependencies.
- **Vector Database**: Utilizes vector database like approach for efficient similarity searches, supporting RAG .
- **Lang Chain**: For integrating LLM operations (LLMOps)
- **Streamlit**: For creating interactive web UI
