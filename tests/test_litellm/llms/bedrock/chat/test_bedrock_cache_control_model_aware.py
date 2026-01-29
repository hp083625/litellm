import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)


class TestBedrockCacheControlModelAware:
    """Test suite for model-aware cache control functionality"""

    def test_get_cache_point_block_with_anthropic_model(self):
        """Test that cache point blocks are created for Anthropic models"""
        config = AmazonConverseConfig()

        # Mock a message block with cache_control
        message_block = {"cache_control": {"type": "ephemeral"}}

        # Test with Anthropic model
        result = config._get_cache_point_block(
            message_block=message_block, block_type="content_block", model="anthropic.claude-3-sonnet-20240229-v1:0"
        )

        # Should return a cache point block for Anthropic models
        assert result is not None
        # The result is a dict with cachePoint key, not an object with cachePoint attribute
        assert "cachePoint" in result

    def test_get_cache_point_block_with_non_anthropic_model(self):
        """Test that cache point blocks are NOT created for non-Anthropic models"""
        config = AmazonConverseConfig()

        # Mock a message block with cache_control
        message_block = {"cache_control": {"type": "ephemeral"}}

        # Test with non-Anthropic model
        result = config._get_cache_point_block(
            message_block=message_block, block_type="content_block", model="qwen.qwen3-coder-480b-a35b-v1:0"
        )

        # Should return None for non-Anthropic models
        assert result is None

    def test_strip_cache_control_fields_for_anthropic_model(self):
        """Test that cache_control fields are preserved for Anthropic models"""
        config = AmazonAnthropicClaudeMessagesConfig()

        # Mock request with cache_control fields
        request = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}]}
            ]
        }

        # Test with Anthropic model - should preserve cache control
        config._strip_cache_control_fields(request, "anthropic.claude-3-sonnet-20240229-v1:0")

        # Cache control fields should be preserved for Anthropic models
        assert "cache_control" in request["messages"][0]["content"][0]

    def test_strip_cache_control_fields_for_non_anthropic_model(self):
        """Test that cache_control fields are stripped for non-Anthropic models"""
        config = AmazonAnthropicClaudeMessagesConfig()

        # Mock request with cache_control fields
        request = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}]}
            ]
        }

        # Test with non-Anthropic model - should strip cache control
        config._strip_cache_control_fields(request, "qwen.qwen3-coder-480b-a35b-v1:0")

        # Cache control fields should be stripped for non-Anthropic models
        assert "cache_control" not in request["messages"][0]["content"][0]

    def test_strip_cache_control_fields_string_content(self):
        """Test cache_control stripping for messages with string content"""
        config = AmazonAnthropicClaudeMessagesConfig()

        # Mock request with string content and cache_control
        request = {"messages": [{"role": "user", "content": "Hello world", "cache_control": {"type": "ephemeral"}}]}

        # Test with non-Anthropic model
        config._strip_cache_control_fields(request, "qwen.qwen3-coder-480b-a35b-v1:0")

        # Cache control field should be stripped
        assert "cache_control" not in request["messages"][0]

    def test_strip_cache_control_fields_mixed_content(self):
        """Test cache_control stripping for mixed content types"""
        config = AmazonAnthropicClaudeMessagesConfig()

        # Mock request with mixed content
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}},
                        {"type": "image", "source": {"data": "abc123"}, "cache_control": {"type": "ephemeral"}},
                    ],
                }
            ]
        }

        # Test with non-Anthropic model
        config._strip_cache_control_fields(request, "qwen.qwen3-coder-480b-a35b-v1:0")

        # All cache control fields should be stripped
        content = request["messages"][0]["content"]
        assert "cache_control" not in content[0]
        assert "cache_control" not in content[1]

    def test_get_cache_point_block_without_model(self):
        """Test that cache point blocks are created when no model is provided"""
        config = AmazonConverseConfig()

        # Mock a message block with cache_control
        message_block = {"cache_control": {"type": "ephemeral"}}

        # Test without model parameter (backward compatibility)
        result = config._get_cache_point_block(message_block=message_block, block_type="content_block")

        # Should return a cache point block for backward compatibility
        assert result is not None
        assert "cachePoint" in result
