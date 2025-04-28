from dotenv import load_dotenv
import os
import re
import pymupdf4llm
from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
cohere_api_key = os.getenv("COHERE_API")

def pdf_to_text(pdf_path):
    """
    Extracts raw text from a PDF using pymupdf4llm, preserving page breaks as '-----'.
    """
    raw_text = pymupdf4llm.to_markdown(pdf_path)
    return raw_text

def clean_pagewise(text):
    """
    Cleans text:
    1. Removes footnotes (numbers stuck to capital letters) ONLY right before page breaks.
    2. Merges broken sentences across page breaks.
    """
    pages = text.split('-----')  # Split text by page breaks
    cleaned_pages = []

    for i, page in enumerate(pages):
        lines = page.strip().split('\n')

        # Step 1: Remove footnote at the end if it matches (number jammed to Capital letter)
        if lines:
            last_line = lines[-1].strip()
            if re.match(r'^\d+[A-Z]', last_line):  # e.g., "2This follows..."
                lines = lines[:-1]  # Drop the last line (footnote)
                print(f"Removed footnote from page {i}")

        cleaned_pages.append('\n'.join(lines))

    # Step 2: Merge sentences split across page breaks
    final_pages = []
    for i in range(len(cleaned_pages) - 1):
        current_page = cleaned_pages[i].rstrip()
        next_page = cleaned_pages[i + 1].lstrip()

        if current_page:
            last_char = current_page[-1]
            # If page ends without sentence-ending punctuation
            if last_char not in '.!?':
                # And next page starts with lowercase or mid-sentence looking
                if next_page and next_page[0].islower():
                    print(f"Merging sentence across page {i} and {i+1}")
                    merged = current_page + " " + next_page
                    cleaned_pages[i + 1] = merged
                    final_pages.append('')  # Current page gets merged into next
                else:
                    final_pages.append(current_page)
            else:
                final_pages.append(current_page)

    # Last page
    final_pages.append(cleaned_pages[-1])

    # Filter out empty merged pages
    final_pages = [page for page in final_pages if page.strip() != '']

    # Recombine pages
    return '\n\n-----\n\n'.join(final_pages)

def format_text(raw_text):
    """
    Sends text to Cohere for strict formatting while preserving all alphanumeric characters.
    """
    prompt = ChatPromptTemplate.from_template("""
    You are a strict text formatting assistant specializing in cleaning up messy PDF extractions.

    You are given raw text extracted from a PDF.
    The text contains these common PDF extraction issues:
    - Words stuck together without spaces (like "Thischapterintroduces")
    - Words broken across lines with hyphens (like "build-\\ning")
    - Incorrect line breaks in the middle of sentences
    - Strange spacing or formatting issues
    - Mathematical notation and symbols that need preservation
    - Footnotes and references that need proper formatting

    **Your task is to ONLY fix formatting issues:**
    1. Separate words that are incorrectly joined (e.g., "Thischapter" → "This chapter")
    2. Rejoin words that are incorrectly hyphenated across lines (e.g., "build-\\ning" → "building")
    3. Fix paragraph breaks and line spacing
    4. Preserve all mathematical notations, symbols, and special characters
    5. Maintain chapter headings, section numbers, and document structure

    **STRICT RULES YOU MUST FOLLOW:**
    - You must NOT change ANY alphanumeric characters - all letters and numbers must remain exactly the same
    - You must NOT add, remove, or reorder any content
    - You must NOT summarize or paraphrase
    - You must NOT add any introductions, notes, explanations, or commentary
    - You must NOT add "---", "**Note:**", or any similar text
    - Start your output immediately with the cleaned text itself
    - Return ONLY the cleaned version of the input text — nothing else

    **CRITICAL CONSTRAINT:**
    The sequence of all alphanumeric characters (A-Z, a-z, 0-9) must be EXACTLY identical between your output and the input.
    This means that if someone were to remove all spaces, punctuation, and special characters from both texts,
    they would get the exact same string of letters and numbers.

    ---

    Here is the raw text:

    {raw_text}
    """)

    llm = ChatCohere(
        model="command-a-03-2025", 
        temperature=0, 
        api_key=cohere_api_key
    )

    chain = prompt | llm

    formatted_text = chain.invoke({
        "raw_text": raw_text
    }).content
    return formatted_text

def remove_non_alphanumeric(text):
    """
    Removes all non-alphanumeric characters for comparison.
    """
    return re.sub(r'[^A-Za-z0-9]', '', text)

def is_alphanumeric_equivalent(text1, text2):
    """
    Checks if two texts have the same alphanumeric sequence.
    """
    cleaned1 = remove_non_alphanumeric(text1)
    cleaned2 = remove_non_alphanumeric(text2)
    print("RAW TEXT:")
    print(cleaned1)
    print("FORMATTED TEXT:")
    print(cleaned2)
    return cleaned1 == cleaned2

def main():
    pdf_path = "../ToolsAndJewels.pdf"  # <-- Update path if needed
    raw_text = pdf_to_text(pdf_path)

    # Smarter cleaning before formatting
    cleaned_raw_text = clean_pagewise(raw_text)

    formatted_text = format_text(cleaned_raw_text)

    attempt = 1
    while not is_alphanumeric_equivalent(cleaned_raw_text, formatted_text):
        print(f"Formatting failed on attempt {attempt}, retrying...")
        formatted_text = format_text(cleaned_raw_text)
        attempt += 1

    print("Final formatted text:")
    print(formatted_text)

if __name__ == '__main__':
    main()
