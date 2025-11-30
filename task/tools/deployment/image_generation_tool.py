from typing import Any

from aidial_sdk.chat_completion import Message
from tools.deployment.base import DeploymentTool
from tools.models import ToolCallParams


class ImageGenerationTool(DeploymentTool):
    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        result: Message = await super()._execute(tool_call_params)

        if result.custom_content and result.custom_content.attachments:
            for attachment in result.custom_content.attachments:
                if attachment.type in ["image/png", "image/jpeg"]:
                    tool_call_params.choice.append_content(
                        f"\n\r![image]({attachment.url})\n\r"
                    )

        if not result.content:
            result.content = "The image has been successfully generated according to request and shown to user!"

        return result

    @property
    def deployment_name(self) -> str:
        return "dall-e-3"

    @property
    def name(self) -> str:
        return "image_generation_tool"

    @property
    def description(self) -> str:
        return """
        Image generation tool.

        Used for generating images based on user request.
        """

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Extensive description of the image that should be generated.",
                },
                "size": {
                    "type": "string",
                    "description": "The size of the generated image.",
                    "enum": ["1024x1024", "1024x1792", "1792x1024"],
                    "default": "1024x1024",
                },
                "style": {
                    "type": "string",
                    "description": "The style of the generated image. Must be one of `vivid` or `natural`. \n- `vivid` causes the model to lean towards generating hyperrealistic and dramatic images. \n- `natural` causes the model to produce more natural, less realistic looking images.",
                    "enum": ["natural", "vivid"],
                    "default": "natural",
                },
                "quality": {
                    "type": "string",
                    "description": "The quality of the image that will be generated. ‘hd’ creates images with finer details and greater consistency across the image.",
                    "enum": ["standard", "hd"],
                    "default": "standard",
                },
            },
            "required": ["prompt"],
        }
