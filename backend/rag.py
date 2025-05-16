import cohere
import numpy as np
import os
from parser import parse_text
from get_atomic_facts import extract_sentences_for_rag

# Setup
co = cohere.ClientV2(os.getenv("COHERE_API"))

# Step 1: Embed the documents
documents = extract_sentences_for_rag(parse_text())[:96]
embed_input = [{"content": [{"type": "text", "text": doc}]} for doc in documents]

doc_emb = co.embed(
    inputs=embed_input,
    model="embed-v4.0",
    output_dimension=1024,
    input_type="search_document",
    embedding_types=["float"],
).embeddings.float
doc_emb = np.array(doc_emb)

# Step 2: Embed the query
query = input("Enter your question: ")
query_input = [{"content": [{"type": "text", "text": query}]}]

query_emb = co.embed(
    inputs=query_input,
    model="embed-v4.0",
    input_type="search_query",
    output_dimension=1024,
    embedding_types=["float"],
).embeddings.float[0]
query_emb = np.array(query_emb)

# Step 3: Compute all cosine similarities
similarities = np.dot(doc_emb, query_emb) / (np.linalg.norm(doc_emb, axis=1) * np.linalg.norm(query_emb))

# Print all documents with similarity scores
print("\n📊 All Documents with Similarity Scores:")
for i, (doc, score) in enumerate(zip(documents, similarities)):
    print(f"{i+1:>2}. Score: {score:.3f} | {doc}")

# Step 4: Filter by threshold
threshold = 0.7
filtered_idxs = [i for i, score in enumerate(similarities) if score >= threshold]
filtered_docs = [documents[i] for i in filtered_idxs]
filtered_scores = [similarities[i] for i in filtered_idxs]

# If no relevant context found, return fallback
if not filtered_docs:
    print("\n💡 Answer:")
    print("I don’t know.")
    exit()

# Step 5: Sort filtered docs by similarity
top_k = 2
sorted_docs = sorted(zip(filtered_docs, filtered_scores), key=lambda x: x[1], reverse=True)[:top_k]

# Step 6: RAG prompt creation
context = "\n".join([doc for doc, _ in sorted_docs])
rag_prompt = f"""Answer the question using only the context provided.
If the answer cannot be found in the context, respond with "I don't know."

Context:
{context}

Question:
{query}

Answer:"""

# Step 7: Generate answer
response = co.generate(
    model="command-r-plus",
    prompt=rag_prompt,
    max_tokens=300,
    temperature=0.3,
)

# Step 8: Print top-k used and the final answer
print("\n📄 Top Facts Used:")
for i, (doc, score) in enumerate(sorted_docs):
    print(f"{i+1}. ({score:.2f}) {doc}")

print("\n💡 Answer:")
print(response.generations[0].text.strip())
