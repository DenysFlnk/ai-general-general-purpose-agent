import os

import uvicorn
from agent import GeneralPurposeAgent
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response
from prompts import SYSTEM_PROMPT
from tools.base import BaseTool
from tools.deployment.image_generation_tool import ImageGenerationTool
from tools.files.file_content_extraction_tool import FileContentExtractionTool
from tools.mcp.mcp_client import MCPClient
from tools.mcp.mcp_tool import MCPTool
from tools.py_interpreter.python_code_interpreter_tool import PythonCodeInterpreterTool
from tools.rag.rag_tool import DocumentCache, RagTool

DIAL_ENDPOINT = os.getenv("DIAL_ENDPOINT", "http://localhost:8080")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")


class GeneralPurposeAgentApplication(ChatCompletion):
    def __init__(self):
        self.tools: list[BaseTool] = []

    async def _get_mcp_tools(self, url: str) -> list[BaseTool]:
        tools: list[BaseTool] = []

        mcp_client: MCPClient = await MCPClient.create(url)
        mcp_tools = await mcp_client.get_tools()

        for tool in mcp_tools:
            tools.append(MCPTool(client=mcp_client, mcp_tool_model=tool))

        return tools

    async def _create_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = []

        tools.append(ImageGenerationTool(endpoint=DIAL_ENDPOINT))
        tools.append(FileContentExtractionTool(endpoint=DIAL_ENDPOINT))
        tools.append(
            RagTool(
                endpoint=DIAL_ENDPOINT,
                deployment_name=DEPLOYMENT_NAME,
                document_cache=DocumentCache.create(),
            )
        )

        interpreter_tool = await PythonCodeInterpreterTool.create(
            mcp_url="http://localhost:8050/mcp",
            tool_name="execute_code",
            dial_endpoint=DIAL_ENDPOINT,
        )
        tools.append(interpreter_tool)

        mcp_tools = await self._get_mcp_tools(url="http://localhost:8051/mcp")
        tools.extend(mcp_tools)

        return tools

    async def chat_completion(self, request: Request, response: Response) -> None:
        if not self.tools:
            self.tools = await self._create_tools()

        with response.create_single_choice() as choice:
            agent = GeneralPurposeAgent(
                endpoint=DIAL_ENDPOINT, system_prompt=SYSTEM_PROMPT, tools=self.tools
            )
            await agent.handle_request(
                deployment_name=DEPLOYMENT_NAME,
                choice=choice,
                request=request,
                response=response,
            )


dial_app = DIALApp()
general_purpose_agent_app = GeneralPurposeAgentApplication()
dial_app.add_chat_completion(
    deployment_name="general-purpose-agent", impl=general_purpose_agent_app
)

if __name__ == "__main__":
    uvicorn.run(dial_app, port=5030, host="0.0.0.0")
