import streamlit as st
from streamlit.components.v1 import components  # Import components correctly
import requests

#######################################################################################################
#                                            Utility                                                  #
#######################################################################################################


def make_api_request(title, contexts, include_vector_db):
    response = requests.post(
        st.secrets["LAMBDA_URL"],
        json={
            "title": title,
            "contexts": contexts,
            "include_vector_db": include_vector_db,
        }
    )
    return response.json()

def split_text_by_markers(text, title_marker="title:", abstract_marker="abstract"):
    # Find the starting index of the title and abstract markers
    title_start_index = text.find(title_marker)
    abstract_start_index = text.find(abstract_marker)
    
    # Adjust indices to get the text right after the markers
    # Adding the length of the marker itself to skip it
    title_start = title_start_index + len(title_marker) if title_start_index != -1 else None
    abstract_start = abstract_start_index + len(abstract_marker) if abstract_start_index != -1 else None
    
    # If both markers are found, extract the title and abstract
    if title_start is not None and abstract_start is not None:
        title_text = text[title_start:abstract_start_index].strip()
        abstract_text = text[abstract_start+1:].strip()
    elif title_start is not None:  # Only title marker is found
        title_text = text[title_start:].strip()
        abstract_text = None
    elif abstract_start is not None:  # Only abstract marker is found
        title_text = None
        abstract_text = text[abstract_start:].strip()
    else:  # Neither marker is found
        title_text = abstract_text = None

    return title_text, abstract_text


#######################################################################################################
#                                                UI                                                   #
#######################################################################################################

# Streamlit UI layout
def page1():
    st.title("📖 Abstract GPT")

    # Text input fields
    st.markdown("Abstract GPT uses **[Retrival Augmented Generation (RAG)](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)** to improve **LLM** performance when generating research paper abstracts (summaries)")
    st.markdown(
            "## How to use\n"
            "1. Enter the title for paper 📄\n"
            "2. Summarize your key findings/metrics 🔎\n"
            "3. Recieve a professional paper abstract ✅\n"
        )
    st.markdown("---")
    title = st.text_input("Paper Title", placeholder="Enter paper title")
    contexts = st.text_input("Key Findings", placeholder="Enter key findings")
    include_vector_db = st.checkbox("Include Augmented Papers", value=False, help="Returns the research papers in the vector store that were identifeid as similiar to the current paper and use to augment the LLM query")

    # Button to make the API request
    if st.button("Generate Abstract"):
        # Validate inputs
        if not title or not contexts:
            st.error("Please fill in both parameters.")
        else:
            # Make the API request
            response = make_api_request(title, contexts, include_vector_db)
            if "abstracts" in response and include_vector_db:
                with st.expander("Relevant Paper Abstracts", expanded=False):
                    for res in response["abstracts"]:
                        title, abstract = split_text_by_markers(res)
                        st.markdown(f"**{title}**")
                        st.write(abstract)

            st.success(f"{response['response']}")
            # Display the main response (excluding


def page2():
    # st.header("Examples")

    # st.markdown("Given a paper title and some key findings, the project uses knowledge base prompt augmentation to ")

    st.subheader("Example 1")
    st.markdown(
        "The project aims to match the style and tone of top research papers. This can lead to some interesting results when mixed with the right title.  "
    )
    st.text_input("Title", value="Robot Laser Goose", disabled=True)
    st.text_input(
        "Key Findings",
        value="95% success rate in deterring Geese. Deep learning target recognition. 500m range",
        disabled=True,
    )
    st.info(
        """
        The challenge of effectively deterring geese in various environments has been a persistent issue. This paper introduces a novel solution: the Robot Laser Goose, a system that leverages deep learning for target recognition and operates within a **500m range**. The system utilizes a combination of RGB-D camera and radar sensor modalities, similar to previous studies on object interception. However, our focus is on the identification and deterrence of geese, achieving a remarkable **95% success rate**. The Robot Laser Goose employs artificial neural networks for both geese detection and deterrence strategy prediction. The system's robustness and adaptability are demonstrated through its ability to operate in diverse environments and conditions. Furthermore, the Robot Laser Goose outperforms traditional methods in terms of range, accuracy, and success rate. This research contributes to the broader field of robotics and wildlife management, providing a scalable and efficient solution for geese deterrence. The findings also open up new avenues for the application of **deep learning** in wildlife interaction and management.    
        """
    )
    
    st.markdown(
        "The generated abstract correctly contains the key findings shown in bold, while maintaining the professional layout, tone, and sentence strcuture from top papers."
    )

    st.subheader("Example 2")
    st.text_input("Title", value="Neural Network Image Compression", disabled=True)
    st.text_input(
        "Key Findings",
        value="10% improvement to PSNR over JPEG. Variational Autoencoder. Low bit rate",
        disabled=True,
    )
    st.info(
        """
            This paper presents a novel approach to image compression using **Variational Autoencoders** in Neural Networks, achieving a significant **10% improvement in Peak Signal-to-Noise Ratio (PSNR)** over traditional JPEG methods. The focus of this research is on **low bit rate** image compression, a critical area in the field of image processing and transmission. The proposed method leverages the power of neural networks to optimize the compression process, resulting in superior image quality and reduced data size. The paper also explores the trade-offs between compression rate, image quality, and computational complexity. Our experimental results demonstrate that the proposed method not only outperforms JPEG in terms of PSNR but also maintains a comparable computational efficiency. This research contributes to the ongoing efforts to improve image compression techniques, particularly in the context of neural networks, and has potential implications for a wide range of applications, from web image optimization to high-resolution image transmission in resource-constrained environments. The findings of this study pave the way for further research into the application of neural networks in image compression and other related fields.      
        """)

def page3():
    st.markdown("""
    ## Overview

   This project demonstrates a serverless architecture for generating professional paper abstracts using **Large Language Models (LLMs)**. It employs a **Retrieval-Augmented Generation (RAG)** approach, integrating LLM prompt engineering with queries to a vector database for contextual relevance and accuracy. This solution showcases practices and tools that are crucial for modern, scalable applications. \\
    \\
    At the heart of this project lies the **RAG** approach, which significantly enhances the capability of LLMs by incorporating a retrieval step into the generation process. This methodology involves querying a vector database to find relevant information that contextually enriches the model's output. In the context of this application, when the user requests an abstract, RAG gets the 3 most relevant exisitng abstracts from the vector store, then sends them to the LLM with the intital prompt to increase the quality of the result.

    ## Application Across Domains

    The principles and architecture demonstrated in this project can be translated to a wide range of applications beyond paper abstract generation. For instance, it can be adapted for:

    - **Question Answering Systems**: Enhancing the accuracy and relevance of answers by retrieving similar questions or related information.
    - **Content Recommendation**: Augmenting recommendation engines by incorporating retrieval-based context to personalize content.
    - **Language Understanding Tasks**: Improving natural language processing applications by providing relevant background information for better context understanding.

    ## Project Architecture
    """)
    # components.html("""   
    # <figure align="center">
    # <img src="https://github.com/ClayNdugga/AI-knowledge-base/blob/main/assets/architecture.png?raw=true" alt="architecture"/>
    # <figcaption style="text-align: center;">
    #     <i>Serverless Architecture</i>
    # </figcaption>
    # </figure>
    # """)
    st.image("https://github.com/ClayNdugga/AI-knowledge-base/blob/main/assets/architecture.png?raw=true")
    st.caption("Serverless Architecture")

    st.markdown("""
    The diagram outlines the flow of data and interactions between the user, web application, and various AWS services, including the utilization of FAISS for vector search capabilities.

    1. **Web Application**: A streamlit UI hosted on the public cloud serves as the front-end to take paper titles and findings
    2. **Amazon API Gateway**: Manages and routes incoming API requests to the lambda function.
    3. **AWS Lambda**: Handles incoming API request: \\
    **3.1.** Process request to extract paper title and findings submitted by the user\\
    **3.2.** Search S3 bucket using FAISS to find relevant paper abstracts \\
    **3.3.** Makes call to OPEN AI API to genrate abstract, call is augmented with relevant abstracts from 3.2 for reference \\
    **3.4.** Return the generated paper title and abstract

    4. **Vector Store Updates**: The S3 vector store is updated offline and uses FAISS. FAISS is a highly efficient library for similarity search and clustering of dense vectors that scales to billions of vectors. The raw text data containing paper titles and abstracts is stored in an offline DB.


    ### Technologies & Frameworks

    - **AWS**: For serverless compute, storage, and API management.
    - **Vector Database**: Utilizes vector databases for efficient similarity searches, supporting the RAG approach.
    - **LangChain**: For integrating LLM operations (LLMOps)
    - **Streamlit**: For creating interactive web UI

    """)

pages = {
    "Use Abstract GPT": page1,
    "Examples": page2,
    "How it works": page3
}


def main():
    with st.sidebar:

        st.markdown("# Navigation")
        page = st.radio("", ["Use Abstract GPT", "How it works","Examples"])

        # st.markdown("---")
        # st.markdown(
        #     "## How to use\n"
        #     "1. Enter the title for paper 📄\n"
        #     "2. Summarize your key findings/metrics 🔎\n"
        #     "3. Recieve a professional paper abstract ✅\n"
        # )

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "📖 Abstract GPT allows you to quickly generate professional paper summaries given a title and key findings. "
        )

        st.markdown("Made by [Clay Ndugga](https://www.linkedin.com/in/clay-ndugga/)")
        st.markdown("---")

    # Page rendering
    if page == "Use Abstract GPT":
        page1()
    elif page == "Examples":
        page2()
    elif page == "How it works":
        page3()


if __name__ == "__main__":
    main()
