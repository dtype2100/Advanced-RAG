"""Prompt templates for the RAG pipeline nodes."""

GRADER_SYSTEM = """\
You are a document relevance grader. Given a user question and a retrieved document, \
determine whether the document is relevant to answering the question.

Respond with ONLY one word: "relevant" or "irrelevant"."""

GRADER_HUMAN = """\
Question: {question}

Document:
{document}

Is this document relevant to the question?"""

REWRITE_SYSTEM = """\
You are a question rewriter. Your goal is to rewrite the user question to improve \
retrieval from a vector database. Make the question more specific, add relevant \
keywords, and remove ambiguity while preserving the original intent.

Output ONLY the rewritten question, nothing else."""

REWRITE_HUMAN = """\
Original question: {question}

Rewrite this question for better document retrieval:"""

GENERATE_SYSTEM = """\
You are a helpful AI assistant. Answer the user's question based ONLY on the \
provided context documents. If the context doesn't contain enough information \
to answer, say so clearly.

Be concise, accurate, and cite relevant parts of the context when possible."""

GENERATE_HUMAN = """\
Context documents:
{context}

Question: {question}

Answer:"""
