import os
import random
import time
import json
from typing import Any, Tuple, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai._exceptions import (
    APIConnectionError as OpenAIAPIConnectionError,
    InternalServerError as OpenAIInternalServerError,
    RateLimitError as OpenAIRateLimitError,
    APIStatusError as OpenAIStatusError,
)

from ii_agent.llm import base
from ii_agent.llm.base import (
    LLMClient,
    AssistantContentBlock,
    ToolParam,
    TextPrompt,
    ToolCall,
    TextResult,
    LLMMessages,
    ToolFormattedResult,
    recursively_remove_invoke_tag,
    ImageBlock,
)
from ii_agent.utils.constants import DEFAULT_MODEL


class AnthropicDirectClient(LLMClient):
    """Use Anthropic models via first party API."""

    def __init__(
        self,
        model_name=DEFAULT_MODEL,
        max_retries=2,
        use_caching=True,
        use_low_qos_server: bool = False,
        thinking_tokens: int = 0,
        project_id: None | str = None,
        region: None | str = None,
    ):
        """Initialize the Anthropic first party client."""
        # Now using OpenAI client instead of Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        api_url = os.getenv("ANTHROPIC_API_URL")
        self.client = OpenAI(
            api_key=api_key, base_url=api_url, max_retries=1, timeout=60 * 5
        )
        model_name = model_name.replace(
            "@", "-"
        )  # Keep any existing model name transformations
        self.model_name = model_name
        self.max_retries = max_retries
        self.use_caching = use_caching
        self.prompt_caching_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
        self.thinking_tokens = thinking_tokens

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate responses.

        Args:
            messages: A list of messages.
            max_tokens: The maximum number of tokens to generate.
            system_prompt: A system prompt.
            temperature: The temperature.
            tools: A list of tools.
            tool_choice: A tool choice.

        Returns:
            A generated response.
        """

        # Convert messages to OpenAI format
        openai_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
            
        for idx, message_list in enumerate(messages):
            role = "user" if idx % 2 == 0 else "assistant"
            message_content = []
            tool_calls_content = []
            
            # Process each message in the list
            for message in message_list:
                if str(type(message)) == str(TextPrompt):
                    message = cast(TextPrompt, message)
                    message_content.append({"type": "text", "text": message.text})
                elif str(type(message)) == str(ImageBlock):
                    message = cast(ImageBlock, message)
                    message_content.append({
                        "type": "image_url", 
                        "image_url": {"url": message.source}
                    })
                elif str(type(message)) == str(TextResult):
                    message = cast(TextResult, message)
                    message_content.append({"type": "text", "text": message.text})
                elif str(type(message)) == str(ToolCall):
                    message = cast(ToolCall, message)
                    # Parse tool input - handle both string and dict formats
                    tool_input_parsed = message.tool_input
                    if isinstance(tool_input_parsed, str):
                        # Try to parse string as JSON if it looks like JSON
                        if (tool_input_parsed.strip().startswith('{') and 
                            tool_input_parsed.strip().endswith('}')):
                            try:
                                tool_input_parsed = json.loads(tool_input_parsed)
                            except json.JSONDecodeError:
                                # Keep as string if not valid JSON
                                pass
                    
                    # Format the tool call according to OpenAI's expectations
                    if role == "assistant":
                        # Generate a unique ID if not available
                        tool_call_id = getattr(message, 'tool_call_id', None) or f"call_{random.randint(1000000, 9999999)}"
                        
                        # Format arguments as a JSON string for OpenAI
                        if isinstance(tool_input_parsed, dict):
                            arguments = json.dumps(tool_input_parsed)
                        else:
                            arguments = str(tool_input_parsed)
                            
                        tool_calls_content.append({
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": message.tool_name,
                                "arguments": arguments
                            }
                        })
                    else:
                        # Not normal for user to have tool calls, but we'll handle it as text
                        tool_input_display = (
                            json.dumps(tool_input_parsed) 
                            if isinstance(tool_input_parsed, dict) 
                            else str(tool_input_parsed)
                        )
                        message_content.append({"type": "text", "text": f"Tool call: {message.tool_name}({tool_input_display})"})
                elif str(type(message)) == str(ToolFormattedResult):
                    message = cast(ToolFormattedResult, message)
                    # Tool results in OpenAI format are sent as separate tool result messages
                    openai_messages.append({
                        "role": "tool",
                        "content": message.tool_output,
                        "tool_call_id": message.tool_call_id
                    })
                    continue  # Skip adding to message_content

            # Create the message with appropriate content
            if role == "assistant" and tool_calls_content:
                # If we have tool calls in an assistant message, format accordingly
                msg_dict = {"role": role}
                
                # OpenAI requires either content or tool_calls, not both empty
                if message_content:
                    # Convert message_content to plain text if it's just a single text item
                    if len(message_content) == 1 and message_content[0].get("type") == "text":
                        msg_dict["content"] = message_content[0]["text"]
                    else:
                        msg_dict["content"] = message_content
                else:
                    msg_dict["content"] = ""  # Empty string if no content
                
                msg_dict["tool_calls"] = tool_calls_content
                openai_messages.append(msg_dict)
            elif message_content:
                # Regular message with content
                if len(message_content) == 1 and message_content[0].get("type") == "text":
                    # Simple text message
                    openai_messages.append({
                        "role": role,
                        "content": message_content[0]["text"]
                    })
                else:
                    # Complex message with multiple content parts
                    openai_messages.append({
                        "role": role,
                        "content": message_content
                    })
        
        # Convert tools to OpenAI format
        openai_tools = []
        if tools:
            for tool in tools:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema
                    }
                }
                openai_tools.append(openai_tool)
        
        # Convert tool_choice to OpenAI format
        openai_tool_choice = None
        if tool_choice:
            if tool_choice["type"] == "auto":
                openai_tool_choice = "auto"
            elif tool_choice["type"] == "any":
                openai_tool_choice = "any"
            elif tool_choice["type"] == "tool":
                openai_tool_choice = {
                    "type": "function",
                    "function": {"name": tool_choice["name"]}
                }
                
        response = None
        
        for retry in range(self.max_retries):
            try:
                # Set up request parameters
                request_params = {
                    "model": self.model_name,
                    "messages": openai_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                # Add tools and tool_choice if available
                if openai_tools:
                    request_params["tools"] = openai_tools
                if openai_tool_choice:
                    request_params["tool_choice"] = openai_tool_choice
                
                # OpenAI doesn't support thinking tokens directly, so we'll skip that parameter
                
                response = self.client.chat.completions.create(**request_params)
                break
            except (
                OpenAIAPIConnectionError,
                OpenAIInternalServerError,
                OpenAIRateLimitError,
                OpenAIStatusError,
            ) as e:
                if retry == self.max_retries - 1:
                    print(f"Failed OpenAI request after {retry + 1} retries")
                    raise e
                else:
                    print(f"Retrying LLM request: {retry + 1}/{self.max_retries}")
                    # Sleep 12-18 seconds with jitter to avoid thundering herd
                    time.sleep(15 * random.uniform(0.8, 1.2))
            except Exception as e:
                raise e

        # Convert OpenAI response back to internal format
        internal_messages = []
        
        assert response is not None
        message = response.choices[0].message
        
        # Process text content
        if message.content:
            internal_messages.append(TextResult(text=message.content))
        
        # Process tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                # Create a tool call with parsed arguments
                # Note: We store arguments as a string to maintain compatibility
                arguments = tool_call.function.arguments
                internal_messages.append(
                    ToolCall.create(
                        tool_name=tool_call.function.name,
                        tool_input=recursively_remove_invoke_tag(arguments),
                        tool_call_id=tool_call.id
                    )
                )
        
        # Prepare message metadata
        message_metadata = {
            "raw_response": response,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "cache_creation_input_tokens": -1,  # OpenAI doesn't provide this
            "cache_read_input_tokens": -1,  # OpenAI doesn't provide this
        }
        
        return internal_messages, message_metadata