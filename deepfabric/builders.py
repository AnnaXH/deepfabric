import logging

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

from .metrics import trace
from .progress import ProgressReporter
from .prompts import VerbalizedPromptBuilder
from .schemas import ChatMessage, Conversation
from .stream_simulator import simulate_stream
from .verbalized import VerbalizedConfig, VerbalizedSampler

if TYPE_CHECKING:
    from .generator import DataSetGeneratorConfig
    from .llm import LLMClient
    from .schemas import ToolRegistry

logger = logging.getLogger(__name__)

# Ceiling on the token budget derived for a distribution call (see
# SingleShotBuilder._verbalized_max_tokens).
VERBALIZED_MAX_TOKENS_CAP = 32000


def conversation_text(conversation: Conversation) -> str:
    """Render a conversation as plain text for similarity comparison.

    Args:
        conversation: Conversation to render.

    Returns:
        The message contents joined by newlines, ignoring system messages (which
        are identical across candidates and would mask real differences).
    """
    return "\n".join(
        message.content
        for message in conversation.messages
        if message.content and message.role != "system"
    )


class ConversationBuilder(ABC):
    """Abstract base class for conversation builders.

    Each builder implements a specific strategy for generating conversations.
    Builders receive typed configuration and dependencies via constructor.

    Attributes:
        llm: LLM client for generation
        config: Typed configuration for the generator
        tool_registry: Optional tool registry for tool-calling conversations
    """

    def __init__(
        self,
        llm: "LLMClient",
        config: "DataSetGeneratorConfig",
        tool_registry: "ToolRegistry | None" = None,
        progress_reporter: ProgressReporter | None = None,
    ):
        """Initialize the conversation builder.

        Args:
            llm: LLM client for making generation requests
            config: Generator configuration (must be Pydantic model)
            tool_registry: Optional tool registry for tool-calling
            progress_reporter: Optional progress reporter for streaming feedback
        """
        self.llm = llm
        self.config = config
        self.tool_registry = tool_registry
        self.progress_reporter = progress_reporter

    @abstractmethod
    async def generate(
        self, topic_prompt: str, error_feedback: str | None = None
    ) -> "Conversation | list[Conversation]":
        """Generate a complete conversation.

        Args:
            topic_prompt: The topic/scenario prompt to generate conversation about
            error_feedback: Optional error message from a previous failed attempt,
                           used to help the model correct its output on retry

        Returns:
            Complete Conversation object (Pydantic model), or a list of them when
            verbalized sampling fans one node out into several candidates. The
            caller applies the configured sizing mode to a list result.

        Raises:
            ValueError: If generation fails validation
        """
        pass


class BuilderType(BaseModel):
    """Type discriminator for builder selection.

    This model ensures type-safe builder selection based on configuration.
    """

    name: str = Field(description="Builder type name")
    requires_tools: bool = Field(default=False, description="Whether this builder requires tools")

    class Config:
        frozen = True


# Builder type constants
SINGLE_SHOT_BUILDER = BuilderType(name="single_shot", requires_tools=False)
SINGLE_TURN_AGENT_BUILDER = BuilderType(name="single_turn_agent", requires_tools=True)


def determine_builder_type(config: "DataSetGeneratorConfig") -> BuilderType:
    """Determine the appropriate builder type from configuration.

    Agent mode is implicit when tools are configured (tool_components or custom_tools).
    Single-turn agent mode is used for tool-calling conversations.

    Args:
        config: Generator configuration (Pydantic model)

    Returns:
        BuilderType indicating which builder to use

    Raises:
        ValueError: If configuration is invalid or unsupported
    """
    # Agent mode is implicit when tools are configured
    has_tools = config.tool_components or config.custom_tools
    if has_tools:
        return SINGLE_TURN_AGENT_BUILDER

    # Non-agent conversations use single-shot generation
    if config.conversation_type in ("basic", "cot"):
        return SINGLE_SHOT_BUILDER

    msg = f"Cannot determine builder type for conversation_type={config.conversation_type}"
    raise ValueError(msg)


class SingleShotBuilder(ConversationBuilder):
    """Builder for simple conversations using single-shot JSON generation.

    This builder generates the entire conversation in one LLM call using
    structured output with JSON schema validation. Suitable for:
    - Basic Q&A conversations
    - Chain-of-thought reasoning without tools
    - Any conversation that can be generated in one pass
    """

    async def generate(
        self, topic_prompt: str, error_feedback: str | None = None
    ) -> Conversation | list[Conversation]:
        """Generate conversation using single LLM call with JSON schema.

        When verbalized sampling is enabled for the generation seam, the single
        call is replaced by a distribution call and several conversations are
        returned - one per selected candidate.

        Args:
            topic_prompt: Topic or scenario to generate conversation about
            error_feedback: Optional error message from a previous failed attempt

        Returns:
            Complete Conversation object, or a list of them under verbalized sampling

        Raises:
            ValueError: If LLM fails to generate valid conversation
        """
        verbalized = self._verbalized_config()
        if verbalized is not None:
            return await self._generate_verbalized(topic_prompt, error_feedback, verbalized)

        # Build the generation prompt
        generation_prompt = self._build_prompt(topic_prompt, error_feedback)

        # Always use non-streaming for reliable structured output
        conversation = await self._generate_non_streaming(generation_prompt)

        # Fire-and-forget: simulate streaming for TUI preview (non-blocking)
        simulate_stream(
            self.progress_reporter,
            conversation.model_dump_json(indent=2),
            source="conversation_gen",
        )

        # Ensure type checker knows this is a Conversation
        conversation = cast(Conversation, conversation)

        return self._finalize(conversation)

    def _finalize(self, conversation: Conversation) -> Conversation:
        """Validate conversation shape and attach the system message.

        Args:
            conversation: Conversation as returned by the LLM.

        Returns:
            The same conversation, with a system message inserted if configured.

        Raises:
            ValueError: If the conversation does not start with a user message.
        """
        # Validate that generated conversation starts with user message
        # (system messages are added by builder, not generated by LLM)
        if conversation.messages and conversation.messages[0].role != "user":
            msg = (
                f"Generated conversation must start with 'user' message, got '{conversation.messages[0].role}'. "
                "System messages are added automatically by the builder."
            )
            raise ValueError(msg)

        # Insert system message if configured
        if self.config.sys_msg:
            conversation.messages.insert(
                0,
                ChatMessage(role="system", content=self.config.dataset_system_prompt or ""),
            )

        return conversation

    def _verbalized_config(self) -> VerbalizedConfig | None:
        """Return the verbalized config for this seam when it is switched on."""
        diversity = getattr(self.config, "diversity", None)
        verbalized = getattr(diversity, "verbalized", None)
        if verbalized is not None and verbalized.enabled:
            return verbalized
        return None

    def _verbalized_max_tokens(self, verbalized: VerbalizedConfig) -> int:
        """Token budget for a distribution call carrying k full candidates.

        One call now has to fit ``k`` complete conversations, so the stage's
        per-sample budget is scaled by ``k`` unless the user set an explicit
        budget. The cap keeps the request inside provider limits.
        """
        if verbalized.max_tokens is not None:
            return verbalized.max_tokens
        return min(self.config.max_tokens * verbalized.k, VERBALIZED_MAX_TOKENS_CAP)

    async def _generate_verbalized(
        self,
        topic_prompt: str,
        error_feedback: str | None,
        verbalized: VerbalizedConfig,
    ) -> list[Conversation]:
        """Generate several conversations for one node via verbalized sampling.

        Args:
            topic_prompt: Topic or scenario to generate conversations about.
            error_feedback: Optional error message from a previous failed attempt.
            verbalized: Verbalized sampling configuration for this seam.

        Returns:
            One conversation per surviving candidate, most-preferred first.

        Raises:
            ValueError: If no candidate survives validation. The message is
                phrased as a validation error so the caller's sample-retry path
                applies, exactly as for a failed single-shot generation.
        """
        base_prompt = self._build_prompt(topic_prompt, error_feedback)
        distribution_prompt = VerbalizedPromptBuilder.build_distribution_prompt(
            base_prompt,
            k=verbalized.k,
            conversation_type=self.config.conversation_type,
            reasoning_style=self.config.reasoning_style,
        )

        sampler = VerbalizedSampler(verbalized, self.llm)
        result = await sampler.sample(
            distribution_prompt,
            Conversation,
            text_of=conversation_text,
            max_retries=self.config.max_retries,
            max_tokens=self._verbalized_max_tokens(verbalized),
            temperature=self.config.temperature,
        )

        conversations: list[Conversation] = []
        invalid = 0
        for candidate in result.selected:
            conversation = cast(Conversation, candidate.payload)
            try:
                conversation = self._finalize(conversation)
            except ValueError as e:
                # Drop the malformed candidate rather than the whole set; the
                # others are independent samples.
                invalid += 1
                logger.debug("Discarding verbalized candidate '%s': %s", candidate.approach, e)
                continue

            if verbalized.keep_probabilities:
                if conversation.metadata is None:
                    conversation.metadata = {}
                conversation.metadata["verbalized"] = {
                    "approach": candidate.approach,
                    "probability": candidate.probability,
                    "normalized_probability": candidate.normalized_probability,
                    "rank": candidate.rank,
                }
            conversations.append(conversation)

        if not conversations:
            msg = (
                "Validation error: no verbalized candidate produced a valid conversation "
                f"({result.generated} verbalised, {invalid} discarded)"
            )
            raise ValueError(msg)

        # Fire-and-forget: simulate streaming for TUI preview (non-blocking)
        simulate_stream(
            self.progress_reporter,
            conversations[0].model_dump_json(indent=2),
            source="conversation_gen",
        )

        self._report_verbalized(result, len(conversations), invalid)
        return conversations

    def _report_verbalized(self, result, kept: int, invalid: int) -> None:
        """Emit realised diversity for one verbalized batch.

        Args:
            result: The sampler's SamplingResult.
            kept: How many candidates became samples.
            invalid: How many candidates failed conversation validation.
        """
        if self.progress_reporter:
            self.progress_reporter.emit_step_complete(
                "Verbalized sampling",
                verbalized_generated=result.generated,
                verbalized_selected=kept,
                verbalized_duplicates_dropped=result.duplicates_dropped,
                verbalized_invalid=invalid,
                verbalized_diversity=result.diversity,
                verbalized_measure=result.measure,
            )
        trace(
            "verbalized_batch",
            {
                "generated": result.generated,
                "selected": kept,
                "duplicates_dropped": result.duplicates_dropped,
                "invalid": invalid,
                "diversity": result.diversity,
                "measure": result.measure,
            },
        )

    async def _generate_non_streaming(self, prompt: str) -> Conversation:
        """Generate conversation using non-streaming LLM call.

        Args:
            prompt: The complete generation prompt

        Returns:
            Generated Conversation object
        """
        return await self.llm.generate_async(
            prompt=prompt,
            schema=Conversation,
            max_retries=self.config.max_retries,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

    def _build_prompt(self, topic_prompt: str, error_feedback: str | None = None) -> str:
        """Build the generation prompt for single-shot generation.

        Args:
            topic_prompt: The topic to generate about
            error_feedback: Optional error message from a previous failed attempt

        Returns:
            Complete prompt string for the LLM
        """
        # Use the generation system prompt as the base
        prompt_parts = [self.config.generation_system_prompt]

        # Add topic/scenario
        prompt_parts.append(f"\nTopic/Scenario: {topic_prompt}")

        # Add error feedback if this is a retry
        if error_feedback:
            prompt_parts.append(
                f"\n\nRETRY: {error_feedback}. Use real values, not null/empty/placeholders."
            )

        # Add any additional instructions
        if self.config.instructions:
            prompt_parts.append(f"\nAdditional Instructions: {self.config.instructions}")

        # Add reasoning-specific guidance based on style
        if self.config.conversation_type == "cot":
            if self.config.reasoning_style == "freetext":
                prompt_parts.append(
                    "\nREASONING FORMAT: Generate natural, conversational reasoning content (string format). "
                    "Show your actual thinking process - explore ideas, consider alternatives, work through the problem. "
                    "Think like a human would: 'Hmm, let me think about this...', 'Wait, that doesn't work...', "
                    "'Actually, if I approach it this way...'. "
                    "DO NOT use numbered steps or structured outlines. "
                    "Use the 'content' field in reasoning as a plain string (not a list)."
                )
            elif self.config.reasoning_style == "agent":
                prompt_parts.append(
                    "\nREASONING FORMAT: Generate structured reasoning steps as a list of ReasoningStep objects. "
                    "Each step should have clear thought and action fields."
                )

        # Add explicit structure requirement
        prompt_parts.append(
            "\nIMPORTANT: Generate the conversation messages array starting with a 'user' message "
            "(the user's question or request), followed by an 'assistant' message (the response). "
            "Do NOT include any 'system' role messages - those are added separately."
        )

        return "\n".join(prompt_parts)


class ConversationBuilderFactory:
    """Factory for creating conversation builders.

    Provides type-safe builder instantiation based on configuration.
    """

    @staticmethod
    def create(
        config: "DataSetGeneratorConfig",
        llm: "LLMClient",
        tool_registry: "ToolRegistry | None" = None,
        progress_reporter: ProgressReporter | None = None,
    ) -> ConversationBuilder:
        """Create the appropriate conversation builder.

        Args:
            config: Generator configuration (Pydantic model)
            llm: LLM client for generation
            tool_registry: Optional tool registry (required for agent builders)
            progress_reporter: Optional progress reporter for streaming feedback

        Returns:
            Appropriate ConversationBuilder instance

        Raises:
            ValueError: If configuration is invalid or builder requirements not met
        """
        builder_type = determine_builder_type(config)

        # Validate tool registry requirement
        if builder_type.requires_tools and tool_registry is None:
            msg = (
                f"Builder type '{builder_type.name}' requires tool_registry but it was not provided"
            )
            raise ValueError(msg)

        # Instantiate appropriate builder
        if builder_type == SINGLE_SHOT_BUILDER:
            return SingleShotBuilder(llm, config, progress_reporter=progress_reporter)
        if builder_type == SINGLE_TURN_AGENT_BUILDER:
            from .builders_agent import SingleTurnAgentBuilder  # noqa: PLC0415

            diversity = getattr(config, "diversity", None)
            verbalized = getattr(diversity, "verbalized", None)
            if verbalized is not None and verbalized.enabled:
                logger.warning(
                    "generation.diversity.verbalized is enabled but tools are configured; "
                    "expression diversity applies to the single-shot path only and will be "
                    "skipped for this agent run"
                )

            return SingleTurnAgentBuilder(
                llm, config, cast("ToolRegistry", tool_registry), progress_reporter
            )
        msg = f"Unknown builder type: {builder_type.name}"
        raise ValueError(msg)
