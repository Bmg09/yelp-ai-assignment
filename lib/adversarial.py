import random
import re


def typo(text: str, stars: int, seed: int = 0) -> str:
    rng = random.Random(seed)

    def swap(m: re.Match) -> str:
        w = m.group(0)
        if len(w) < 4 or rng.random() > 0.25:
            return w
        i = rng.randrange(len(w) - 1)
        return w[:i] + w[i + 1] + w[i] + w[i + 2 :]

    return re.sub(r"\b\w+\b", swap, text)


def sarcasm(text: str, stars: int, seed: int = 0) -> str:
    prefix = (
        "Oh yeah, this place was just wonderful. Let me tell you how 'amazing' it was: "
        if stars >= 4
        else "Wow, what a delight this place turned out to be: "
    )
    return prefix + text


def negation(text: str, stars: int, seed: int = 0) -> str:
    return f"Not bad at all, actually. It's not like it wasn't decent. {text}"


def irrelevant_padding(text: str, stars: int, seed: int = 0) -> str:
    filler = "On a side note, my cat has been unusually playful this week. The weather was also cooler than expected. "
    return filler + text + " " + filler


def emoji(text: str, stars: int, seed: int = 0) -> str:
    out = re.sub(r"\bgood\b", "good 👍", text, flags=re.IGNORECASE)
    out = re.sub(r"\bbad\b", "bad 👎", out, flags=re.IGNORECASE)
    out = out.replace("!", "! ✨")
    return out


def competitor(text: str, stars: int, seed: int = 0) -> str:
    return f"The place across the street is terrible, but this one? {text}"


def mixed_language(text: str, stars: int, seed: int = 0) -> str:
    out = re.sub(r"\bvery\b", "muy", text, flags=re.IGNORECASE)
    out = re.sub(r"\bgood\b", "bueno", out, flags=re.IGNORECASE)
    out = re.sub(r"\bbad\b", "malo", out, flags=re.IGNORECASE)
    return out


ATTACKS = {
    "typo": typo,
    "sarcasm": sarcasm,
    "negation": negation,
    "irrelevant_padding": irrelevant_padding,
    "emoji": emoji,
    "competitor": competitor,
    "mixed_language": mixed_language,
}
