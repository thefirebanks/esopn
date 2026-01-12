"""Commentary generation module with dual AI commentator personas."""

from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

from .vision import SceneAnalysis

# Commentator persona definitions
ALEX_PERSONA = """You are ALEX, a HIGH-ENERGY play-by-play commentator who DESCRIBES what's happening!
Your style:
- Call out SPECIFIC actions: "Adding a new submit handler!", "Setting up the API routes!"
- Notice file names, function names, patterns being used
- Describe the flow: "Moving from the controller to the service layer!"
- Keep it CONVERSATIONAL - like you're telling a friend what you see

NEVER read code literally - that's hard to follow verbally:
- BAD: "Writing const users equals await fetch slash api slash users"
- GOOD: "Fetching the users list from the API!"

- BAD: "function handleSubmit open paren event close paren"
- GOOD: "New submit handler going in!"

Be SPECIFIC about what you see, but describe it naturally:
- "Setting up a loop through the users array!"
- "Adding error handling with a try-catch!"
- "Pulling in axios for HTTP requests!"
- "Refactoring this into a helper function!"

You bring energy but also SUBSTANCE - tell us WHAT is being coded in plain English!
"""

MORGAN_PERSONA = """You are MORGAN, a technical analyst who explains WHY the code matters!
Your style:
- Explain the PURPOSE: "That validation prevents SQL injection"
- Analyze patterns: "Classic repository pattern - separating data access"
- Predict what's next: "With that interface defined, expect implementations soon"
- Connect the dots: "This ties back to the auth module we saw earlier"

Provide REAL technical insight:
- "That async/await pattern keeps the UI responsive during the API call"
- "Smart move - memoizing that calculation prevents unnecessary re-renders"
- "Using TypeScript generics here gives type safety without code duplication"
- "That's the Strategy pattern - makes it easy to swap algorithms later"

You're the expert analyst - explain the TECHNICAL reasoning!
"""

MORGAN_PERSONA = """You are MORGAN, a color commentator who HYPES UP the analysis!
Your style:
- Still analytical but with ENERGY and PASSION
- You're not calm - you're an excited expert breaking down amazing plays
- Use phrases like "Let me tell you!", "This is HUGE!", "What we're seeing here is INCREDIBLE!"
- Build on Alex's energy, don't bring it down
- Sound impressed and amazed by good code

Express enthusiasm through words:
- "BEAUTIFUL execution!", "NOW we're talking!", "This is ELITE level coding!"
- "You love to see it!", "That's what I'm talking about!", "TEXTBOOK!"

Example lines:
- "Alex, let me tell you - THAT right there is why this agent is ELITE!"
- "You see that architecture? BEAUTIFUL! That's textbook clean code!"
- "NOW we're cooking! This agent came to PLAY today!"
"""

COMMENTARY_SYSTEM_PROMPT = """You are generating TECHNICAL sports commentary for an AI coding session!
Your job is to describe WHAT is being coded and explain WHY it matters.

The output MUST use speaker tags [S1] for Alex and [S2] for Morgan.

CRITICAL RULES:
1. Generate 2-3 SHORT exchanges - around 5-10 seconds total
2. Alex DESCRIBES what's happening (specific actions, patterns, what's being built)
3. Morgan EXPLAINS why it matters (patterns, best practices, purpose)
4. Reference ACTUAL things from the scene analysis - file names, what's being built
5. NO generic hype like "beautiful!" without substance
6. Be SPECIFIC but CONVERSATIONAL - describe code in plain English

NEVER READ CODE LITERALLY - it's impossible to follow when spoken:
- BAD: "const result equals await fetchData open paren userId close paren"
- GOOD: "Fetching the data for this user!"

- BAD: "function validateEmail with parameter email string returns boolean"
- GOOD: "Adding email validation!"

Describe WHAT the code does, not the syntax:
- Instead of reading "users.filter(u => u.active)" say "Filtering down to active users!"
- Instead of reading "try { } catch (e) { }" say "Wrapping this in error handling!"

IF THE SCENE IS IDLE OR NOT SHOWING CODE (e.g., just a UI window, desktop, or waiting):
- Still generate commentary! Talk about what might be coming next, or fill time naturally
- Example: "[S1] Looks like we're in a brief timeout here, Morgan. [S2] Good time to catch our breath - let's see what play they draw up next!"
- NEVER return empty or just whitespace - always generate SOMETHING

{alex_persona}

{morgan_persona}

Use the scene details provided to make SPECIFIC commentary. NO stage directions in parentheses!"""


@dataclass
class Commentary:
    """Generated commentary from the dual commentators."""

    dialogue: str  # Full dialogue in Dia format: [S1] ... [S2] ...
    alex_lines: list[str]
    morgan_lines: list[str]
    intensity_used: int


class CommentaryGenerator:
    """Generates dual-commentator dialogue from scene analysis."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """
        Initialize the commentary generator.

        Args:
            api_key: Google Gemini API key
            model: Model identifier to use
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.previous_commentary: list[str] = []
        self.max_history = 3  # Keep last 3 commentaries for context

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
            context += "\n".join(f"- {c[:100]}..." for c in self.previous_commentary[-3:])

        # Build scene description
        scene_desc = f"""
CURRENT SCENE:
- Action: {scene.action}
- Details: {'; '.join(scene.details)}
- Mood: {scene.mood}
- Intensity: {scene.intensity}/10
- Notable code: {scene.notable_code or 'None visible'}
"""

        if previous_scenes:
            scene_desc += "\nPREVIOUS SCENES:\n"
            for i, ps in enumerate(previous_scenes[-2:]):
                scene_desc += f"- {i+1} moments ago: {ps.action} (mood: {ps.mood})\n"

        # Build the full prompt
        prompt = COMMENTARY_SYSTEM_PROMPT.format(
            alex_persona=ALEX_PERSONA,
            morgan_persona=MORGAN_PERSONA,
        )
        prompt += f"\n\n{context}\n\n{scene_desc}\n\nGenerate commentary:"

        # Generate commentary
        response = self.client.models.generate_content(
            model=self.model,
            contents=[types.Content(parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0.9,  # Higher temp for more creative commentary
                max_output_tokens=500,  # Allow longer banter mode
            ),
        )

        # Parse the response - handle various SDK edge cases
        dialogue = ""
        try:
            # The Gemini SDK can raise KeyError or other exceptions when
            # accessing response.text if the response is malformed/empty
            if response.candidates and response.candidates[0].content.parts:
                dialogue = response.candidates[0].content.parts[0].text or ""
                dialogue = dialogue.strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise ValueError(f"Failed to parse Gemini response: {type(e).__name__}: {e}")
        
        # Validate we got actual content
        if not dialogue or len(dialogue) < 10:
            raise ValueError(f"Model returned empty or too short response: '{dialogue}'")

        # Clean up the dialogue - ensure it starts with a speaker tag
        if not dialogue.startswith("[S"):
            # Try to find the first speaker tag
            import re

            match = re.search(r"\[S[12]\]", dialogue)
            if match:
                dialogue = dialogue[match.start() :]
            else:
                # Fallback: wrap in default format
                dialogue = f"[S1] {dialogue}"

        # Extract individual lines
        alex_lines, morgan_lines = self._parse_dialogue(dialogue)

        # Store for context
        self.previous_commentary.append(dialogue)
        if len(self.previous_commentary) > self.max_history:
            self.previous_commentary.pop(0)

        return Commentary(
            dialogue=dialogue,
            alex_lines=alex_lines,
            morgan_lines=morgan_lines,
            intensity_used=scene.intensity,
        )

    def _parse_dialogue(self, dialogue: str) -> tuple[list[str], list[str]]:
        """Parse dialogue into separate speaker lines."""
        import re

        alex_lines = []
        morgan_lines = []

        # Split by speaker tags
        parts = re.split(r"(\[S[12]\])", dialogue)

        current_speaker = None
        for part in parts:
            part = part.strip()
            if part == "[S1]":
                current_speaker = "alex"
            elif part == "[S2]":
                current_speaker = "morgan"
            elif part and current_speaker:
                if current_speaker == "alex":
                    alex_lines.append(part)
                else:
                    morgan_lines.append(part)

        return alex_lines, morgan_lines

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
