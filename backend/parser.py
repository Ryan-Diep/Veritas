from dotenv import load_dotenv
import os
import re
import pymupdf4llm
from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
cohere_api_key = os.getenv("COHERE_API")

contractions = { 
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

def pdf_to_text(pdf_path):
    raw_text = pymupdf4llm.to_markdown(pdf_path)
    return remove_contractions(raw_text.lower())

def format_text(raw_text):

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
        api_key=cohere_api_key)

    chain = prompt | llm

    formatted_text = chain.invoke({
        "raw_text" : raw_text
    }).content
    return formatted_text

def remove_contractions(text):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in contractions) + r')\b')
    return pattern.sub(lambda match: contractions[match.group(0)], text)

def remove_non_alphanumeric(text):
    return re.sub(r'[^A-Za-z0-9]', '', text)

def is_alphanumeric_equivalent(text1, text2):
    return remove_non_alphanumeric(text1) == remove_non_alphanumeric(text2)


def parse_text():
    pdf_path = "../ToolsAndJewels.pdf"
    raw_text = re.sub(r'(\w)\n', r'\1 ', pdf_to_text(pdf_path))
    formatted_text = format_text(raw_text)
    # while(not is_alphanumeric_equivalent(raw_text, formatted_text)):
    #     formatted_text = format_text(raw_text)
    return formatted_text

if __name__ == '__main__':
    print(parse_text())