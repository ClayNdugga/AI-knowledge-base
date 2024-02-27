import streamlit as st
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
    st.markdown("Abstract GPT uses **Knowledge Base Embedding's** stored in a **vector DB** to augment LLM queries with relevant research paper abstracts to create more accurate responses.")
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
    st.markdown("DIAGRM")

pages = {
    "Use Abstract GPT": page1,
    "Examples": page2,
    "How it work's": page3
}


def main():
    with st.sidebar:

        st.markdown("# Navigation")
        page = st.radio("", ["Use Abstract GPT", "How it work's","Examples"])

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
    elif page == "How it work's":
        page3()


if __name__ == "__main__":
    main()
