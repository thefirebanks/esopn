"""Vision analysis module using Google Gemini for screenshot understanding."""

from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

from .capture import Screenshot

# System prompt for analyzing coding screenshots
VISION_SYSTEM_PROMPT = """You are an expert at analyzing screenshots of AI coding agents at work.
Extract SPECIFIC technical details that commentators can use.

Focus on extracting:
1. SPECIFIC code being written (function names, variable names, imports)
2. File names and paths visible
3. What ACTION is happening (writing function X, fixing bug Y, adding import Z)
4. Any error messages or test results (exact text if visible)
5. Patterns being used (React hooks, async/await, class inheritance, etc.)

IMPORTANT - IGNORE THESE (they are NOT real errors):
- IDE/editor warnings about imports (grayed out squiggles, "could not be resolved")
- Type checker warnings (pyright, mypy, typescript errors in the problems panel)
- These are just tooling noise, NOT actual code problems!

Be SPECIFIC and CONCRETE. Extract actual names and code snippets.
Output a JSON object with these fields:
- action: SPECIFIC description like "Writing async function fetchUsers in api.ts" (not just "writing code")
- details: SPECIFIC bullet points like "Adding useState hook for loading state", "Importing axios from node_modules"
- mood: One of "triumph", "tension", "progress", "struggle", "neutral"
- intensity: 1-10 scale of how exciting/noteworthy this moment is
- notable_code: Actual code snippets, function names, or error messages visible (be specific!)"""


@dataclass
class SceneAnalysis:
    """Structured analysis of a screenshot."""

    action: str
    details: list[str]
    mood: str
    intensity: int
    notable_code: Optional[str] = None
    raw_response: Optional[str] = None


class VisionAnalyzer:
    """Analyzes screenshots using Gemini vision model."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """
        Initialize the vision analyzer.

        Args:
            api_key: Google Gemini API key
            model: Model identifier to use
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def analyze(self, screenshot: Screenshot, context: Optional[str] = None) -> SceneAnalysis:
        """
        Analyze a screenshot and return structured scene information.

        Args:
            screenshot: Screenshot to analyze
            context: Optional context from previous analyses

        Returns:
            SceneAnalysis with structured information about the screenshot
        """
        # Build the prompt
        prompt = VISION_SYSTEM_PROMPT
        if context:
            prompt += f"\n\nContext from previous moments:\n{context}"

        prompt += "\n\nAnalyze this screenshot and respond with JSON:"

        # Get base64 image
        image_data = screenshot.to_base64(max_size=(1280, 960))

        # Create the content with image
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/png",
                                data=image_data,
                            )
                        ),
                        types.Part(text=prompt),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=500,
            ),
        )

        # Parse the response
        return self._parse_response(response.text)

    def _parse_response(self, text: str) -> SceneAnalysis:
        """Parse the model response into a SceneAnalysis object."""
        import json
        import re

        # Try to extract JSON from the response
        # Handle markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback: create from raw text
                return SceneAnalysis(
                    action="Analysis in progress",
                    details=[text[:200] if text else "Unable to analyze"],
                    mood="neutral",
                    intensity=5,
                    raw_response=text,
                )

        try:
            data = json.loads(json_str)
            return SceneAnalysis(
                action=data.get("action", "Unknown action"),
                details=data.get("details", []),
                mood=data.get("mood", "neutral"),
                intensity=min(10, max(1, int(data.get("intensity", 5)))),
                notable_code=data.get("notable_code"),
                raw_response=text,
            )
        except (json.JSONDecodeError, ValueError):
            return SceneAnalysis(
                action="Analysis in progress",
                details=[text[:200] if text else "Unable to analyze"],
                mood="neutral",
                intensity=5,
                raw_response=text,
            )


async def analyze_screenshot_async(
    screenshot: Screenshot,
    api_key: str,
    model: str = "gemini-2.0-flash",
    context: Optional[str] = None,
) -> SceneAnalysis:
    """Async convenience function for analyzing a screenshot."""
    analyzer = VisionAnalyzer(api_key=api_key, model=model)
    return analyzer.analyze(screenshot, context)
