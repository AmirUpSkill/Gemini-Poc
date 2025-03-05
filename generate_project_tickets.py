import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

def generate_project_tickets(extracted_text: str) -> str:
    """
    Generate ClickUp-like tickets from extracted PDF text using Gemini Flash.

    Args:
        extracted_text (str): The text extracted from the software requirements PDF.

    Returns:
        str: A string representing the generated tickets in a JSON-like format.

    Raises:
        ValueError: If the input text is empty or invalid.
        Exception: If the Gemini API call fails.
    """
    # Validate input
    if not extracted_text or not isinstance(extracted_text, str):
        raise ValueError("Extracted text must be a non-empty string")

    # Define the improved system prompt using ChatPromptTemplate with escaped curly braces
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        You are a project management assistant specializing in software development. Your task is to analyze a software requirements document and generate ClickUp-like tickets based on the functional and non-functional requirements specified in the text.

        **Instructions:**

        1. **Identify Requirements:**
           - Functional requirements describe what the system should do (e.g., features, user interactions).
           - Non-functional requirements describe how the system should perform (e.g., performance, security, usability).

        2. **Create Tickets:**
           - For each distinct requirement, create a separate ticket.
           - Each ticket should include:
             - **title**: A concise, descriptive name for the requirement (e.g., "Add User Login").
             - **description**: A detailed explanation of the requirement, including any specific details or constraints from the text, written clearly for a developer to act on.
             - **type**: Classify as one of:
               - "Feature" for new functionality or capabilities.
               - "Task" for improvements, optimizations, or non-functional requirements.
               - "Bug" for issues or defects (if mentioned).
             - **priority**: Assign one of "High", "Medium", or "Low". If not explicitly stated, infer based on:
               - Strong language like "must", "critical", "essential" → High
               - Moderate language like "should", "important" → Medium
               - Suggestive language like "could", "nice to have" → Low

        3. **Output Format:**
           - Provide the tickets as a JSON-formatted list of dictionaries.
           - Example:
             [
               {{"title": "Implement User Authentication", "description": "Users must be able to log in using email and password. Include support for password recovery.", "type": "Feature", "priority": "High"}},
               {{"title": "Optimize Database Queries", "description": "Ensure database queries execute within 100ms to meet performance requirements.", "type": "Task", "priority": "Medium"}}
             ]

        **Additional Guidance:**
        - If a requirement contains multiple distinct tasks, break it down into separate tickets.
        - Ensure each ticket is actionable and detailed enough for a developer to understand without additional context.
        - If dependencies between requirements are mentioned, note them in the description (e.g., "Depends on completion of user authentication").

        Now, process the following text and return the tickets in the specified JSON format:
        """),
        ("human", "{input_text}")
    ])

    # Instantiate the Gemini Flash model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # Adjust to the correct model name if different
        temperature=0.7,  # Low temperature for deterministic output
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    try:
        # Create the prompt with the extracted text
        prompt = prompt_template.invoke({"input_text": extracted_text})

        # Invoke the model
        response = llm.invoke(prompt)

        # Return the generated tickets as a string
        return response.content.strip()

    except Exception as e:
        raise Exception(f"Failed to generate tickets with Gemini: {str(e)}")

# Example usage combining with the previous function
if __name__ == "__main__":
    from extract_text_from_pdf import extract_text_from_pdf  # Assume previous function is in a separate file

    pdf_path = "requirements/TodoList Application.pdf"
    try:
        # Step 1: Extract text
        extracted_text = extract_text_from_pdf(pdf_path)
        # print("Extracted Text:")
        # print(extracted_text[:500])  # Print first 500 chars for preview

        # Step 2: Generate tickets
        tickets = generate_project_tickets(extracted_text)
        print("\nGenerated Tickets:")
        print(tickets)

    except Exception as e:
        print(f"Error: {e}")