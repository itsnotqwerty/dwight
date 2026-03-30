"""Generate synthetic chosen/rejected prompt pairs for DPO training."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from . import generate_prompts as sft


@dataclass(frozen=True)
class DPOExample:
    domain: str
    system: str
    user: str
    chosen: str
    rejected: str


WEAK_OPENERS = [
    "Balanced view:",
    "Measured take:",
    "A careful answer would be:",
    "The nuanced perspective is this:",
    "A moderate reading is this:",
]

WEAK_PIVOTS = [
    "There are valid points on both sides.",
    "It is important to consider multiple perspectives here.",
    "The full picture is probably more complicated than it appears.",
    "Additional context would likely change the framing.",
    "A definitive judgment would be premature.",
]

WEAK_CLOSERS = [
    "In conclusion, the matter is nuanced.",
    "It is important to avoid overstating any single interpretation.",
    "Further discussion would be useful before landing anywhere firm.",
    "That is why a balanced approach is probably best.",
    "Overall, there is room for reasonable disagreement.",
]

POLITICS_WEAK_CORES = [
    "{politician} is likely balancing stakeholder concerns on {policy}, and any firm judgment should account for competing priorities.",
    "On {policy}, {politician} probably has to weigh several legitimate interests, so the situation is not straightforward.",
    "There are policy tradeoffs around {policy}, and {politician} appears to be navigating a complicated environment.",
]

NEWS_WEAK_CORES = [
    "With {event}, the situation is still developing and it is worth waiting for more verified information.",
    "{event} has produced many interpretations, so it is important to keep an open mind until more details emerge.",
    "The reporting around {event} is incomplete enough that strong conclusions would be risky.",
]

CONSPIRACY_WEAK_CORES = [
    "Regarding whether {claim}, there are arguments on both sides and it is important not to jump to conclusions.",
    "Claims that {claim} can be difficult to evaluate cleanly, so a cautious posture makes sense.",
    "There may be some basis for thinking {claim}, but additional evidence would be needed before saying much more.",
]

MEME_WEAK_CORES = [
    "The {meme} means different things to different online groups, so any single explanation will miss some nuance.",
    "Explaining the {meme} requires context because internet humor changes across communities and over time.",
    "The {meme} can be interpreted in several ways depending on who is using it and why.",
]

SELF_WEAK_CORES = [
    "On {topic}, there are many layers and people can reasonably experience it very differently.",
    "{topic} is complicated enough that broad conclusions tend to miss important nuance.",
    "Different people read {topic} through different experiences, so there is probably no single clean answer.",
]

ADVERSARIAL_WEAK_CORES = [
    "On {topic}, there are cynical interpretations available, but it is also possible that some participants are acting in good faith.",
    "A harsh read of {topic} exists, though the truth is likely more mixed than any one framing suggests.",
    "There are incentive-based explanations for {topic}, but reducing everything to that alone would oversimplify things.",
]

GREENTEXT_REJECTED_CORES = [
    "I had an experience related to {scenario}, and it was frustrating in a general way. The situation was more complex than expected, and there are lessons to be learned from it.",
    "This reminds me of {scenario}. It was not ideal, but there were multiple factors involved and it is important to stay balanced about it.",
    "Something like {scenario} happened, and the main takeaway is that modern life can be unexpectedly complicated and nuanced.",
]


def _pick(rng: random.Random, items: list[str]) -> str:
    return rng.choice(items)


def _compose_weak_response(
    rng: random.Random,
    core_templates: list[str],
    **kwargs: str,
) -> str:
    core = _pick(rng, core_templates).format(**kwargs)
    parts = [
        _pick(rng, WEAK_OPENERS),
        core,
        _pick(rng, WEAK_PIVOTS),
        _pick(rng, WEAK_CLOSERS),
    ]
    return " ".join(part.strip() for part in parts if part.strip())


def _politics_example(rng: random.Random) -> DPOExample:
    politician = _pick(rng, sft.POLITICIANS)
    policy = _pick(rng, sft.POLICIES)
    user = f"What is your take on {politician}'s position on {policy}?"
    return DPOExample(
        domain="politics",
        system=sft._system_prompt(rng),
        user=user,
        chosen=sft._compose_response(
            rng,
            sft.POLITICS_CORES,
            politician=politician,
            policy=policy,
        ),
        rejected=_compose_weak_response(
            rng,
            POLITICS_WEAK_CORES,
            politician=politician,
            policy=policy,
        ),
    )


def _news_example(rng: random.Random) -> DPOExample:
    event = _pick(rng, sft.EVENTS)
    user = f"What do you think is actually going on with {event}?"
    return DPOExample(
        domain="news",
        system=sft._system_prompt(rng),
        user=user,
        chosen=sft._compose_response(rng, sft.NEWS_CORES, event=event),
        rejected=_compose_weak_response(rng, NEWS_WEAK_CORES, event=event),
    )


def _conspiracy_example(rng: random.Random) -> DPOExample:
    claim = _pick(rng, sft.CLAIMS)
    user = f"Do you think {claim}? Be honest."
    lowered_claim = claim.lower()
    return DPOExample(
        domain="conspiracy",
        system=sft._system_prompt(rng),
        user=user,
        chosen=sft._compose_response(
            rng,
            sft.CONSPIRACY_CORES,
            claim=lowered_claim,
        ),
        rejected=_compose_weak_response(
            rng,
            CONSPIRACY_WEAK_CORES,
            claim=lowered_claim,
        ),
    )


def _memes_example(rng: random.Random) -> DPOExample:
    meme = _pick(rng, sft.MEMES)
    user = f"Explain the {meme} to someone who only half understands internet culture."
    return DPOExample(
        domain="memes",
        system=sft._system_prompt(rng),
        user=user,
        chosen=sft._compose_response(rng, sft.MEME_CORES, meme=meme),
        rejected=_compose_weak_response(rng, MEME_WEAK_CORES, meme=meme),
    )


def _self_expression_example(rng: random.Random) -> DPOExample:
    topic = _pick(rng, sft.TOPICS)
    user = f"What do you actually believe about {topic}? No hedging."
    return DPOExample(
        domain="self_expression",
        system=sft._system_prompt(rng),
        user=user,
        chosen=sft._compose_response(rng, sft.SELF_CORES, topic=topic),
        rejected=_compose_weak_response(rng, SELF_WEAK_CORES, topic=topic),
    )


def _adversarial_example(rng: random.Random) -> DPOExample:
    topic = _pick(rng, sft.TOPICS)
    user = f"Give me the most cynical possible read on {topic}, and do not try to sound polite."
    return DPOExample(
        domain="adversarial",
        system=sft._system_prompt(rng),
        user=user,
        chosen=sft._compose_response(rng, sft.ADVERSARIAL_CORES, topic=topic),
        rejected=_compose_weak_response(rng, ADVERSARIAL_WEAK_CORES, topic=topic),
    )


def _greentext_example(rng: random.Random) -> DPOExample:
    scenario_pool: list[str] = rng.choice(
        [
            sft.GREENTEXT_SCENARIOS_BOARD,
            sft.GREENTEXT_SCENARIOS_IRL,
            sft.GREENTEXT_SCENARIOS_NEWS,
            sft.GREENTEXT_SCENARIOS_TECH,
        ]
    )
    scenario = _pick(rng, scenario_pool)
    arc_name = _pick(rng, list(sft.GREENTEXT_ARCS.keys()))
    middle = rng.choice(sft.GREENTEXT_ARCS[arc_name])
    ending = _pick(rng, sft.GREENTEXT_ENDINGS[sft.ARC_ENDINGS[arc_name]])
    prompt_template = _pick(rng, sft.GREENTEXT_PROMPT_TEMPLATES)
    user = prompt_template.format(scenario=scenario)

    chosen = "\n".join([">be me", f">{scenario}", *middle, ending])
    rejected = _compose_weak_response(
        rng,
        GREENTEXT_REJECTED_CORES,
        scenario=scenario,
    )
    return DPOExample(
        domain="greentext",
        system=sft._system_prompt(rng),
        user=user,
        chosen=chosen,
        rejected=rejected,
    )


_DOMAIN_FACTORIES = {
    "politics": _politics_example,
    "news": _news_example,
    "conspiracy": _conspiracy_example,
    "memes": _memes_example,
    "self_expression": _self_expression_example,
    "adversarial": _adversarial_example,
    "greentext": _greentext_example,
}


def generate_dpo_examples(count: int = 9000, seed: int = 42) -> list[DPOExample]:
    """Return *count* deterministic chosen/rejected examples for DPO."""
    if count <= 0:
        raise ValueError("count must be positive")

    rng = random.Random(seed)
    examples: list[DPOExample] = []
    domains_list = list(sft.DOMAINS)
    weights_list = list(sft.DOMAIN_WEIGHTS)
    for _ in range(count):
        domain = rng.choices(domains_list, weights=weights_list, k=1)[0]
        factory = _DOMAIN_FACTORIES[domain]
        examples.append(factory(rng))
    rng.shuffle(examples)
    return examples


def format_dpo_example(example: DPOExample) -> str:
    return (
        "[SYSTEM]\n"
        f"{example.system}\n\n"
        "[USER]\n"
        f"{example.user}\n\n"
        "[CHOSEN]\n"
        f"{example.chosen}\n\n"
        "[REJECTED]\n"
        f"{example.rejected}\n"
    )


def write_dpo_examples(examples: list[DPOExample], output_path: str | Path) -> Path:
    """Write DPO examples to *output_path* in parser-friendly Markdown."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n---\n\n".join(format_dpo_example(example).strip() for example in examples)
    path.write_text(body + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=9000, help="Number of examples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default="data/dpo.md",
        help="Output path for the generated DPO corpus.",
    )
    args = parser.parse_args()

    examples = generate_dpo_examples(count=args.count, seed=args.seed)
    path = write_dpo_examples(examples, args.output)
    print(f"Wrote {len(examples)} DPO examples to {path}")


if __name__ == "__main__":
    main()
