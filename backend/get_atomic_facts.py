import spacy
from typing import List
import re

def extract_sentences_for_rag(text: str) -> List[str]:
    """
    Extract atomic facts as a simple list of sentences ready for RAG embeddings
    
    Args:
        text: Input text to process
        
    Returns:
        List of sentences
    """
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_lg")
    except:
        try:
            nlp = spacy.load("en_core_web_md")
        except:
            nlp = spacy.load("en_core_web_sm")
    
    # Process the text
    doc = nlp(text)
    
    # Store extracted sentences
    sentences = []
    
    # Process each sentence to extract facts
    for sent in doc.sents:
        sent_doc = nlp(sent.text.strip())
        
        # Extract subject-verb-object triples
        for token in sent_doc:
            # Find subject-verb pairs
            if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ in ("VERB", "AUX"):
                # Get the subject
                subject = get_span_text(token)
                
                # Get the verb phrase
                verb_parts = [token.head.text]
                for child in token.head.children:
                    if child.dep_ in ("aux", "auxpass", "neg", "prt"):
                        verb_parts.append(child.text)
                verb_parts.sort(key=lambda x: token.doc.text.find(x))
                verb = " ".join(verb_parts)
                
                # Find objects for this verb
                for obj_token in token.head.children:
                    if obj_token.dep_ in ("dobj", "pobj", "attr", "iobj"):
                        # Get the object
                        obj = get_span_text(obj_token)
                        
                        # Create sentence
                        sentence = f"{subject} {verb} {obj}."
                        sentence = clean_sentence(sentence)
                        sentences.append(sentence)
                
                # Handle prepositional phrases
                for prep in token.head.children:
                    if prep.dep_ == "prep" and prep.pos_ == "ADP":
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj":
                                # Get object of preposition
                                prep_obj = get_span_text(pobj)
                                
                                # Create sentence with preposition
                                sentence = f"{subject} {verb} {prep.text} {prep_obj}."
                                sentence = clean_sentence(sentence)
                                sentences.append(sentence)
    
    return sentences

def get_span_text(token):
    """Extract the full noun phrase for a token"""
    if token.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj", "iobj", "attr"):
        span_tokens = [t.text for t in token.subtree 
                      if not (t.dep_ in ("prep", "punct") and t.head != token)]
        span_tokens.sort(key=lambda x: token.doc.text.find(x))
        return " ".join(span_tokens)
    return token.text

def clean_sentence(sentence):
    """Clean and normalize extracted sentences"""
    # Normalize spaces
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    # Ensure proper capitalization
    sentence = sentence[0].upper() + sentence[1:]
    
    # Ensure proper ending punctuation
    if not sentence.endswith(('.', '!', '?')):
        sentence += '.'
    
    return sentence