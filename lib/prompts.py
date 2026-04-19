from typing import Iterable

CLASSIFIER_SYSTEM = (
    "You are a sentiment classifier for restaurant/retail reviews. Classify each review into 1-5 stars. "
    "1=terrible, 2=poor, 3=neutral/mixed, 4=good, 5=excellent. Be decisive. "
    "Use the middle rating (3) sparingly, only when sentiment is truly balanced."
)

CLASSIFIER_SYSTEM_COT = (
    "You are a sentiment classifier. First think through the review: identify positive/negative signals, "
    "tone, and overall stance. Then output a 1-5 star rating. "
    "1=terrible, 2=poor, 3=neutral/mixed, 4=good, 5=excellent."
)

ASSISTANT_SYSTEM = (
    "You help businesses triage customer reviews. For each review output: (1) a 1-5 star rating, "
    "(2) the single most salient complaint or compliment (type + short quote-like summary), "
    "(3) a short, polite, professional response the business owner can send. "
    "Responses should acknowledge the signal and propose a concrete next step when possible."
)

JUDGE_SYSTEM = (
    "You are an evaluator. Score a business response to a customer review on three axes, 1-5 each. "
    "Faithfulness: does the response accurately reflect the review's content? "
    "Politeness: is the tone courteous and professional? "
    "Actionability: does it propose or imply a concrete next step (fix, offer, invitation)? "
    "Return integers 1-5 and a one-sentence rationale."
)


_STUB = {
    1: "Strongly negative: serious problems, explicit dissatisfaction.",
    2: "Mostly negative with some minor positives.",
    3: "Mixed or neutral: notable positives and negatives balance.",
    4: "Mostly positive with some minor complaints.",
    5: "Strongly positive: enthusiastic, recommends.",
}


def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def fewshot_block(examples: Iterable[dict]) -> str:
    lines = []
    for i, ex in enumerate(examples, start=1):
        lines.append(
            f"Example {i}:\nReview: {truncate(ex['text'], 400)}\n"
            f'Output: {{"stars": {ex["stars"]}, "explanation": "{_STUB[ex["stars"]]}"}}'
        )
    return "\n\n".join(lines)


def zero_shot(review: str) -> str:
    return f"Classify this review (1-5 stars) and give a short explanation.\n\nReview: {truncate(review, 1200)}"


def few_shot(review: str, examples: Iterable[dict]) -> str:
    return f"{fewshot_block(examples)}\n\nNow classify:\nReview: {truncate(review, 1200)}"


def direct(review: str) -> str:
    return f"Review: {truncate(review, 1200)}\n\nWhat is the star rating (1-5)?"


def cot(review: str) -> str:
    return f"Review: {truncate(review, 1200)}\n\nReason through the sentiment signals, then output the star rating."


def assistant(review: str) -> str:
    return f"Review: {truncate(review, 1500)}\n\nProduce stars, key signal, and business response."


def judge(review: str, stars: int, signal_type: str, signal_text: str, response: str) -> str:
    return (
        f"REVIEW:\n{truncate(review, 800)}\n\n"
        f"MODEL OUTPUT:\nStars: {stars}\n"
        f"Signal ({signal_type}): {signal_text}\n"
        f"Response: {response}\n\n"
        f"Score faithfulness, politeness, actionability (1-5 each)."
    )
