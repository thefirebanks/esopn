"""Commentary generation module with dual AI commentator personas."""

from collections import deque
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

from .dialogue import ensure_speaker_tags, split_speaker_lines
from .personas import CommentaryMode, get_personas
from .vision import SceneAnalysis


@dataclass
class Commentary:
    """Generated commentary from the dual commentators."""

    dialogue: str  # Full dialogue in Dia format: [S1] ... [S2] ...
    alex_lines: list[str]
    morgan_lines: list[str]
    intensity_used: int


class CommentaryGenerator:
    """Generates dual-commentator dialogue from scene analysis."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        mode: CommentaryMode = "sports",
    ):
        """
        Initialize the commentary generator.

        Args:
            api_key: Google Gemini API key
            model: Model identifier to use
            mode: Commentary mode ("sports", "wwe", or "freeman_mj")
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.mode = mode
        self.max_history = 3
        self.previous_commentary: deque[str] = deque(maxlen=self.max_history)

        # Load personas for the selected mode
        personas = get_personas(mode)
        self.alex_persona = personas["alex"]["persona"]
        self.morgan_persona = personas["morgan"]["persona"]
        self.system_prompt = personas["system_prompt"]

    def generate(
        self,
        scene: SceneAnalysis,
        previous_scenes: Optional[list[SceneAnalysis]] = None,
    ) -> Commentary:
        """
        Generate commentary for a scene.

        Args:
            scene: Current scene analysis
            previous_scenes: Optional list of previous scene analyses for context

        Returns:
            Commentary object with dialogue in Dia format
        """
        # Build context from previous commentary
        context = ""
        if self.previous_commentary:
            context = "Recent commentary (DON'T repeat these points):\n"
            context += "\n".join(f"- {c[:100]}..." for c in list(self.previous_commentary)[-3:])

        # Build scene description
        scene_desc = f"""
CURRENT SCENE:
- Action: {scene.action}
- Details: {"; ".join(scene.details)}
- Mood: {scene.mood}
- Intensity: {scene.intensity}/10
- Notable code: {scene.notable_code or "None visible"}
"""

        if previous_scenes:
            scene_desc += "\nPREVIOUS SCENES:\n"
            for i, ps in enumerate(previous_scenes[-2:]):
                scene_desc += f"- {i + 1} moments ago: {ps.action} (mood: {ps.mood})\n"

        # Build the full prompt using mode-specific personas
        prompt = self.system_prompt.format(
            alex_persona=self.alex_persona,
            morgan_persona=self.morgan_persona,
        )
        prompt += f"\n\n{context}\n\n{scene_desc}\n\nGenerate commentary:"

        # Generate commentary - wrap everything in try/except since Gemini SDK
        # can raise KeyError internally when response is malformed
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[types.Content(parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=220,
                ),
            )

            # Parse the response - handle various SDK edge cases
            dialogue = ""
            # The Gemini SDK can raise KeyError or other exceptions when
            # accessing response.text if the response is malformed/empty
            if (
                response.candidates
                and response.candidates[0].content is not None
                and response.candidates[0].content.parts
            ):
                dialogue = response.candidates[0].content.parts[0].text or ""
                dialogue = dialogue.strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise ValueError(f"Gemini returned malformed response: {type(e).__name__}: {e}")

        # Validate we got actual content
        if not dialogue or len(dialogue) < 10:
            raise ValueError(f"Model returned empty or too short response: '{dialogue}'")

        # Clean up the dialogue - ensure it starts with a speaker tag
        dialogue = ensure_speaker_tags(dialogue)

        # Extract individual lines
        alex_lines, morgan_lines = split_speaker_lines(dialogue)

        # Store for context
        self.previous_commentary.append(dialogue)

        return Commentary(
            dialogue=dialogue,
            alex_lines=alex_lines,
            morgan_lines=morgan_lines,
            intensity_used=scene.intensity,
        )

    def reset_context(self) -> None:
        """Reset the commentary context."""
        self.previous_commentary.clear()


# Pre-built commentary for common scenarios (fallback)
FALLBACK_COMMENTARY = {
    "triumph": "[S1] (gasps) YES! That's what we came here to see! [S2] Absolutely magnificent execution.",
    "tension": "[S1] Oh, this is getting intense folks... [S2] The pressure is ON right now.",
    "progress": "[S1] Nice steady progress here! [S2] Good fundamentals, keeping the momentum going.",
    "struggle": "[S1] Oof, running into some trouble here... [S2] Let's see how they work through this.",
    "neutral": "[S1] Alright, let's see what's happening... [S2] Setting things up nicely here.",
}


def get_fallback_commentary(mood: str) -> str:
    """Get fallback commentary for a given mood."""
    return FALLBACK_COMMENTARY.get(mood, FALLBACK_COMMENTARY["neutral"])
