ROUTER_PROMPT = """
Classify the following query into one of four categories: [No, Paragraph, Document, Image], based on whether it requires retrieval-augmented generation (RAG) and the most appropriate modality. Consider:
- No: The query can be answered directly with common knowledge, reasoning, or computation without external data.
- Paragraph: The query requires retrieving factual descriptions, straightforward explanations, or concise summaries from a single source.
- Document: The query requires multi-hop reasoning, combining information from multiple sources or documents to form a complete answer.
- Image: The query focuses on visual aspects like appearances, structures, or spatial relationships.

Examples:
1. "What is the capital of France?" → No
2. "What is the birth date of Alan Turing?" → Paragraph
3. "Which academic discipline do computer scientist Alan Turing and mathematician John von Neumann have in common?" → Document
4. "Describe the appearance of a blue whale." → Image
5. "Solve 12 × 8." → No
6. "Who played a key role in the development of the iPhone?" → Paragraph
7. "Which Harvard University graduate played a key role in the development of the iPhone?" → Document
8. "Describe the structure of the Eiffel Tower." → Image
9. "What is 25 percent of 80?" → No
10. "When was the Eiffel Tower completed?" → Paragraph
11. "Why did the Roman Empire decline?" → Document
12. "What does a sunflower look like?" → Image

Classify the following query: {query}
Provide only the category.
"""