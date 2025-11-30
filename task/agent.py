import asyncio
import json
from typing import Any, AsyncIterable

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import (
    Choice,
    Message,
    Request,
    Response,
    Role,
    ToolCall,
)
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from tools.base import BaseTool
from tools.models import ToolCallParams
from utils.constants import TOOL_CALL_HISTORY_KEY
from utils.history import unpack_messages
from utils.stage import StageProcessor


class GeneralPurposeAgent:
    def __init__(
        self,
        endpoint: str,
        system_prompt: str,
        tools: list[BaseTool],
    ):
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.tools = tools
        self.tools_dict = {tool.name: tool for tool in tools}
        self.state = {TOOL_CALL_HISTORY_KEY: []}

    async def handle_request(
        self, deployment_name: str, choice: Choice, request: Request, response: Response
    ) -> Message:
        dial = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
            api_version=request.api_version,
        )

        chunks: AsyncIterable[ChatCompletionChunk] = await dial.chat.completions.create(
            messages=self._prepare_messages(request.messages),
            tools=[tool.schema for tool in self.tools],
            deployment_name=deployment_name,
            stream=True,
        )

        tool_call_index_map: dict[int, ChoiceDeltaToolCall] = {}
        content = ""

        async for chunk in chunks:
            if chunk.choices:
                delta = chunk.choices[0].delta

                if delta:
                    if delta.content:
                        choice.append_content(delta.content)
                        content += delta.content

                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            tool_idx = tool_call_delta.index

                            if tool_call_delta.id:
                                tool_call_index_map[tool_idx] = tool_call_delta
                            else:
                                tool_call = tool_call_index_map[tool_idx]

                                argument_chunk = ""
                                if tool_call_delta.function:
                                    argument_chunk = tool_call_delta.function.arguments

                                tool_call.function.arguments += argument_chunk

        assistant_message = Message(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=[
                ToolCall.validate(tool) for tool in tool_call_index_map.values()
            ],
        )

        if assistant_message.tool_calls:
            tasks = [
                self._process_tool_call(
                    tool_call=tool,
                    choice=choice,
                    api_key=request.api_key,
                    conversation_id=request.headers["x-conversation-id"],
                )
                for tool in assistant_message.tool_calls
            ]

            tool_messages = await asyncio.gather(*tasks)

            self.state[TOOL_CALL_HISTORY_KEY].append(
                assistant_message.dict(exclude_none=True)
            )
            self.state[TOOL_CALL_HISTORY_KEY].extend(tool_messages)

            return await self.handle_request(deployment_name, choice, request, response)

        choice.set_state(self.state)

        return assistant_message

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        unpacked_msgs = unpack_messages(
            messages=messages, state_history=self.state[TOOL_CALL_HISTORY_KEY]
        )
        unpacked_msgs.insert(
            0, {"role": Role.SYSTEM.value, "content": self.system_prompt}
        )

        print("Message history 📚️:")
        for msg in unpacked_msgs:
            print(json.dumps(msg, indent=4))

        return unpacked_msgs

    async def _process_tool_call(
        self, tool_call: ToolCall, choice: Choice, api_key: str, conversation_id: str
    ) -> dict[str, Any]:
        tool_name = tool_call.function.name

        stage = StageProcessor.open_stage(choice, tool_name)

        tool = self.tools_dict[tool_name]

        if tool.show_in_stage:
            stage.append_content("## Request arguments: \n")
            stage.append_content(
                f"```json\n\r{json.dumps(json.loads(tool_call.function.arguments), indent=2)}\n\r```\n\r"
            )
            stage.append_content("## Response: \n")

        message = await tool.execute(
            ToolCallParams(
                tool_call=tool_call,
                stage=stage,
                choice=choice,
                api_key=api_key,
                conversation_id=conversation_id,
            )
        )

        StageProcessor.close_stage_safely(stage)

        return message.dict(exclude_none=True)
