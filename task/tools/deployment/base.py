import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterable

from aidial_client import AsyncDial
from aidial_client.types.chat import ChatCompletionChunk
from aidial_sdk.chat_completion import CustomContent, Message, Role
from tools.base import BaseTool
from tools.models import ToolCallParams


class DeploymentTool(BaseTool, ABC):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        pass

    @property
    def tool_parameters(self) -> dict[str, Any]:
        return {}

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        args: dict = json.loads(tool_call_params.tool_call.function.arguments)
        promt = args.pop("prompt")

        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_version="2025-01-01-preview",
            api_key=tool_call_params.api_key,
        )

        stream: AsyncIterable[
            ChatCompletionChunk
        ] = await dial_client.chat.completions.create(
            messages=[{"role": Role.USER, "content": promt}],
            deployment_name=self.deployment_name,
            stream=True,
            extra_body={"custom_fields": {"configuration": {**args}}},
            **self.tool_parameters,
        )

        content = ""
        attachments = []

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta

            if delta:
                if delta and delta.content:
                    tool_call_params.stage.append_content(delta.content)
                    content += delta.content

                if delta.custom_content and delta.custom_content.attachments:
                    attachments = delta.custom_content.attachments

                    for attachment in attachments:
                        tool_call_params.stage.add_attachment(
                            type=attachment.type,
                            title=attachment.title,
                            data=attachment.data,
                            url=attachment.url,
                            reference_url=attachment.reference_url,
                            reference_type=attachment.reference_type,
                        )

        return Message(
            role=Role.TOOL,
            content=content,
            custom_content=CustomContent(attachments=attachments),
            tool_call_id=tool_call_params.tool_call.id,
        )
