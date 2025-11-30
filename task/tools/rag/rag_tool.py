import json
from typing import Any

import faiss
import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tools.base import BaseTool
from tools.models import ToolCallParams
from tools.rag.document_cache import DocumentCache
from utils.dial_file_conent_extractor import DialFileContentExtractor

_SYSTEM_PROMPT = """
You are a RAG-powered assistant that assists users with their questions.
            
## Structure of User message:
`USER QUESTION` - The user's actual question.
`RAG CONTEXT` - Retrieved documents relevant to the query.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""


class RagTool(BaseTool):
    """
    Performs semantic search on documents to find and answer questions based on relevant content.
    Supports: PDF, TXT, CSV, HTML.
    """

    def __init__(
        self, endpoint: str, deployment_name: str, document_cache: DocumentCache
    ):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.document_cache = document_cache
        self.transformer = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "rag_tool"

    @property
    def description(self) -> str:
        return """
    Performs semantic search on documents to find and answer questions based on relevant content.
    Use this tool when user wants to perform search on large document.
    Supports: PDF, TXT, CSV, HTML.
    """

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for in the document",
                },
                "file_url": {"type": "string", "description": "File URL"},
            },
            "required": ["request", "file_url"],
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        args = json.loads(tool_call_params.tool_call.function.arguments)
        request = args["request"]
        file_url = args["file_url"]

        stage = tool_call_params.stage
        stage.append_content("## Request arguments: \n")
        stage.append_content(f"**Request**: {request}\n\r")
        stage.append_content(f"**File URL**: {file_url}\n\r")

        cache_document_key = f"{tool_call_params.conversation_id}-{file_url}"

        cache = self.document_cache.get(cache_document_key)

        if cache:
            index, chunks = cache
        else:
            text_content = DialFileContentExtractor(
                self.endpoint, tool_call_params.api_key
            ).extract_text(file_url)

            if not text_content:
                stage.append_content("**File content is not found!**")
                return "File content is not found"

            chunks = self.text_splitter.split_text(text_content)
            embeddings = self.transformer.encode(chunks)
            index = faiss.IndexFlatL2(384)
            index.add(np.array(embeddings).astype("float32"))
            self.document_cache.set(cache_document_key, index, chunks)

            query_embedding = self.transformer.encode([request]).astype("float32")
            _, indices = index.search(query_embedding, k=3)

            retrieved_chunks = [chunks[idx] for idx in indices[0]]
            augmented_prompt = self.__augmentation(request, retrieved_chunks)
            stage.append_content("## RAG Request: \n")
            stage.append_content(f"```text\n\r{augmented_prompt}\n\r```\n\r")
            stage.append_content("## Response: \n")

            dial_client = AsyncDial(
                base_url=self.endpoint,
                api_version="2025-01-01-preview",
                api_key=tool_call_params.api_key,
            )

            stream = await dial_client.chat.completions.create(
                messages=[
                    {"role": Role.SYSTEM, "content": _SYSTEM_PROMPT},
                    {"role": Role.USER, "content": augmented_prompt},
                ],
                deployment_name=self.deployment_name,
                stream=True,
            )

            content = ""

            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        tool_call_params.stage.append_content(delta.content)
                        content += delta.content

            return content

    def __augmentation(self, request: str, chunks: list[str]) -> str:
        rag_context = "\n".join(chunks)
        return f"USER QUESTION: {request}\n\n RAG CONTEXT: {rag_context}"
