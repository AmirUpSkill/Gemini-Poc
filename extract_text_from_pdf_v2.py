import os
from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file given its file path

    Args :
        pdf_path (str) : The file path to the PDF document

    Returns :
        str : The extracted text content from the PDF

    Raises:
        FileNotFoundError: If the PDF file does not exist at the specified path.
        Exception: If there's an issue loading or processing the PDF.
    """
    # Validate the file path
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"The file '{pdf_path}' does not exist or is not a file.")
    try:
        # Load the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        # Extract text from each page
        full_text = "\n".join(doc.page_content for doc in documents)
        return full_text
    except Exception as e:
        raise Exception(f"An error occurred while loading or processing the PDF: {str(e)}")

# Example usage
if __name__ == "__main__":
    pdf_path = "requirements/TodoList Application.pdf"
    try:
        extracted_text = extract_text_from_pdf(pdf_path)
        print("Extracted Text:")
        print(extracted_text)  # Print entire text
    except Exception as e:
        print(f"Error: {e}")