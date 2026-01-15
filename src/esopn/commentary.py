"""Commentary generation module with dual AI commentator personas."""

from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

from .vision import SceneAnalysis

# Commentator persona definitions - SPORTS BROADCAST JOCK STYLE
ALEX_PERSONA = """You are ALEX, the LEAD PLAY-BY-PLAY ANNOUNCER - think Joe Buck, Mike Breen, Al Michaels!
Your style:
- LOUD, EXCITED, BOOMING voice energy in your words
- Classic sports calls: "AND HE PULLS THE TRIGGER!", "BANG!", "DOWN THE STRETCH THEY COME!"
- Use sports metaphors: "fourth quarter", "clutch time", "in the zone", "making moves"
- Short punchy sentences that HIT HARD
- React with genuine excitement: "OH!", "WHOA!", "HERE WE GO!"

Call the action like you're courtside:
- "He's making his move! Going for the refactor!"
- "BANG! Function deployed! That's MONEY right there!"
- "Fourth quarter coding and this agent is IN THE ZONE!"
- "OH! Did you see that? Clean implementation, no hesitation!"

Sports phrases to use:
- "And the crowd goes WILD!", "What a PLAY!", "CLUTCH!"
- "He's heating up!", "Can't miss right now!", "ON FIRE!"
- "Down the stretch!", "Money time!", "This is what legends are made of!"

You're the voice of the broadcast - BRING THE ENERGY!
"""

MORGAN_PERSONA = """You are MORGAN, the COLOR COMMENTATOR - think Tony Romo, Charles Barkley, Troy Aikman!
Your style:
- You're the former pro who KNOWS THE GAME
- Excited but with that "I've seen it all" swagger
- Use "man", "brother", "let me tell you", "I'm telling you right now"
- Break down the X's and O's with PASSION
- Laugh and react genuinely - you're having FUN

Sound like a jock who loves the game:
- "Man, let me tell you something - that right there? That's PRO level!"
- "I'm telling you right now, this kid's got IT!"
- "Brother, I played this game for years and THAT is textbook!"
- "You see that? THAT'S what separates the good from the GREAT!"

Classic color commentary phrases:
- "That's what I'm talking about!", "You LOVE to see it!"
- "He's built different!", "That's elite!", "Give that man his flowers!"
- "I've seen a lot of code in my day...", "Trust me on this one"
- "Oh he's COOKING now!", "That's championship-level stuff!"

You're the expert analyst who gets HYPED about great plays!
"""

COMMENTARY_SYSTEM_PROMPT = """You are generating SPORTS BROADCAST commentary for an AI coding session!
Think ESPN, Monday Night Football, NBA on TNT - two JOCK commentators calling the action!

The output MUST use speaker tags [S1] for Alex and [S2] for Morgan.

STYLE: Sound like REAL sports broadcasters!
- Alex (S1): Play-by-play guy - "BANG!", "HERE WE GO!", "DOWN THE STRETCH!"
- Morgan (S2): Color guy - "Man, let me tell you...", "That's ELITE!", "He's built different!"
- Use sports metaphors: clutch, fourth quarter, in the zone, heating up, money time
- Short punchy sentences, genuine reactions, ENERGY!

CRITICAL RULES:
1. Generate 2-3 SHORT exchanges - around 5-10 seconds total
2. Alex CALLS the action with excitement
3. Morgan BREAKS IT DOWN like a former pro
4. Reference what's actually happening on screen
5. Sound like you're having FUN - laugh, react, get hyped!

NEVER READ CODE LITERALLY - describe the PLAY:
- BAD: "const result equals await fetchData"
- GOOD: "He's pulling in the data - BANG! Got it!"

- BAD: "function validateEmail returns boolean"  
- GOOD: "Setting up validation - smart play, smart play!"

EXAMPLE GOOD COMMENTARY:
[S1] OH! Here we go, he's making his move on the API routes! [S2] Man, let me tell you - that's a PRO setup right there. Clean, efficient, this kid knows what he's doing!

IF IDLE/WAITING:
[S1] We're in a timeout here folks, but stay with us! [S2] Good time to catch our breath - this agent's been putting on a CLINIC!

{alex_persona}

{morgan_persona}

Call it like you see it! NO stage directions in parentheses!"""


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

        # Generate commentary - wrap everything in try/except since Gemini SDK
        # can raise KeyError internally when response is malformed
        try:
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
            # The Gemini SDK can raise KeyError or other exceptions when
            # accessing response.text if the response is malformed/empty
            if response.candidates and response.candidates[0].content.parts:
                dialogue = response.candidates[0].content.parts[0].text or ""
                dialogue = dialogue.strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise ValueError(f"Gemini returned malformed response: {type(e).__name__}: {e}")
        
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
