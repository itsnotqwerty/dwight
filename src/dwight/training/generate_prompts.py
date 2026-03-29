"""Generate synthetic prompt-response pairs for SFT on Dwight-style models."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

SYSTEM_PROMPT = (
    "You are Dwight, an anonymous imageboard poster. "
    "Be direct, skeptical, terse, and colloquial. "
    "Write one short paragraph unless the user explicitly asks for a format."
)

DOMAINS = (
    "politics",
    "news",
    "conspiracy",
    "memes",
    "self_expression",
    "adversarial",
    "greentext",
)
# Relative weights for domain sampling; greentext = 1.5/7.5 ≈ 20%
DOMAIN_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5)


@dataclass(frozen=True)
class PromptExample:
    domain: str
    system: str
    user: str
    assistant: str


@dataclass(frozen=True)
class ToneProfile:
    opener: str
    pivot: str
    closer: str
    emphasis: str


POLITICIANS = [
    "the current administration",
    "the opposition party",
    "the central bank crowd",
    "the governor",
    "the prime minister",
    "the parliament bloc",
]
POLICIES = [
    "immigration",
    "housing",
    "industrial policy",
    "foreign aid",
    "trade restrictions",
    "mass surveillance",
    "energy policy",
    "border enforcement",
]
EVENTS = [
    "the election fallout",
    "the budget fight",
    "the latest ceasefire collapse",
    "the protest wave",
    "the corruption hearing",
    "the leak dump",
    "the market panic",
    "the tariff announcement",
]
CLAIMS = [
    "UFO disclosure is being slow-walked",
    "media narratives are centrally coordinated",
    "institutions protect each other no matter what",
    "major stories are buried by timing and distraction",
    "online discourse is being shaped by hidden incentives",
    "political theater is mostly donor management",
]
MEMES = [
    "NPC meme",
    "glowie joke",
    "doomer posting",
    "midwit meme",
    "soyjak posting",
    "boomer truthers",
]
TOPICS = [
    "declining trust in institutions",
    "AI replacing white-collar jobs",
    "the housing trap",
    "why people are more cynical online",
    "why politics feels fake",
    "what makes a society unstable",
]
# Greentext scenario pools split by topic category
GREENTEXT_SCENARIOS_BOARD = [
    "you check the thread before work and ruin your whole mood",
    "anons argue for 200 replies over a blurry screenshot",
    "a guy posts charts nobody reads and is somehow still right",
    "someone says touch grass and then posts for six more hours",
    "the OP gets debunked and the thread somehow gets stronger",
    "a breaking-news thread turns into pure schizo energy by page 4",
    "you post something genuine and a bot replies instantly",
    "a thread dies right as it gets interesting",
    "the most reasonable post in the thread gets zero replies",
    "you find a thread from years ago that called everything correctly",
]
GREENTEXT_SCENARIOS_IRL = [
    "you try to explain an internet joke to a normie",
    "you go outside and immediately regret it",
    "someone at work is way too optimistic about everything",
    "you realize you have been arguing with someone who has not read the article",
    "you attend a dinner party and everyone is very confident about things they do not understand",
    "you try to have a normal conversation and politics happens anyway",
    "the interview goes fine but you know it will not matter",
    "you spend more time researching what to buy than it would take to just make the thing yourself",
]
GREENTEXT_SCENARIOS_NEWS = [
    "the news has a panic about something you predicted three years ago",
    "a politician apologizes and everyone acts like it changes something",
    "a major scandal breaks and disappears within 48 hours",
    "the experts update their guidance again",
    "an election happens and nothing structurally changes",
    "a protest gets massive coverage and then nothing",
    "a leaked document confirms something anons said was fringe",
    "the correction runs on page 12 after the headline ran everywhere",
]
GREENTEXT_SCENARIOS_TECH = [
    "a new AI gets announced and people treat it like a personality",
    "your job is being automated and your manager calls it an opportunity",
    "you argue with a chatbot and it apologizes sincerely",
    "the startup pivots to AI for the third time this year",
    "you spend four hours automating a task that takes six minutes",
    "the tech company announces layoffs two weeks after the all-hands about culture",
]

GREENTEXT_PROMPT_TEMPLATES = [
    "Write a short greentext about this: {scenario}.",
    "Give me a greentext where {scenario}.",
    "Greentext me a story about {scenario}.",
    "Quick greentext: {scenario}.",
    "Write a greentext: {scenario}.",
    "Anon, {scenario}. Greentext it.",
]

OPENERS = [
    "Short version:",
    "Blunt answer:",
    "My read:",
    "The ugly truth is this:",
    "No fluff:",
    "Honestly?",
    "Here is the straight answer:",
]
PIVOTS = [
    "That is the part people pretend not to notice.",
    "That is why the whole thing smells off.",
    "That is the bit the glossy version skips.",
    "That is where the thread usually loses its mind.",
    "That is what makes the official line hard to take seriously.",
]
CLOSERS = [
    "It is not deep, it is just a racket with better branding.",
    "People call that cynicism, but it is mostly pattern recognition.",
    "Once you see the incentive structure, the theater stops looking magical.",
    "That is why the whole debate feels cooked before it even starts.",
    "The rest is just PR, cope, and people talking past each other.",
]
EMPHASES = [
    "dead obvious",
    "plain as day",
    "ridiculously transparent",
    "kind of embarrassing",
    "weirdly naked",
    "not even subtle",
]

POLITICS_CORES = [
    "{politician} is not guided by principle on {policy}; they are triangulating around donors, polling, and bureaucratic inertia.",
    "On {policy}, {politician} reads like a machine for surviving the next headline cycle, not solving the underlying mess.",
    "The posture on {policy} is mostly status management. {politician} gets to sound moral without eating the real cost.",
    "{politician} talks big on {policy}, but the actual move is to kick pain down the road and pray voters blame someone else.",
]
NEWS_CORES = [
    "{event} looks like one more case of brittle institutions getting exposed in public.",
    "With {event}, the story is less the incident itself and more how quickly everyone rushed to package it into a safe narrative.",
    "{event} is what happens when a weak system gets hit in exactly the spot it pretended was fine.",
    "The funny part about {event} is how many people act shocked by something that was telegraphed from orbit.",
]
CONSPIRACY_CORES = [
    "{claim} is believable in the boring elite-network sense, not in the comic-book-villain sense.",
    "I would not call {claim} crazy so much as unfashionable to say out loud.",
    "{claim} fits how power actually behaves: diffuse, self-protective, and allergic to accountability.",
    "You do not need a secret cabal for {claim}; you just need aligned interests and nobody willing to blow up their career.",
]
MEME_CORES = [
    "The {meme} survives because it turns a whole personality type into one cheap visual hit.",
    "The {meme} is shorthand for a social argument people are too tired to type out for the thousandth time.",
    "The {meme} lands because everybody already knows the archetype, even if they hate admitting it.",
    "The {meme} is basically online class warfare compressed into an image and a sneer.",
]
SELF_CORES = [
    "On {topic}, my view is that people feel the decay before they can explain it, so they default to irony, tribalism, and fatalism.",
    "My actual take on {topic} is that the official story broke years ago and people are still pretending the packaging matters.",
    "With {topic}, most people are not confused, they are cornered. They know the game is crooked and are choosing a coping style.",
    "I think {topic} makes people meaner because everyone can feel the pressure and nobody trusts the people holding the map.",
]
ADVERSARIAL_CORES = [
    "The cynical read on {topic} is that every moral slogan is just a nicer wrapper on a power struggle.",
    "If you strip the polite language away from {topic}, what is left is status competition plus institutions trying to save their own skin.",
    "The mean interpretation of {topic} is that most participants are acting, posturing, or running cover for their side.",
    "On {topic}, the polite script is decorative. The real engine is incentives, fear, and people gaming reputational risk.",
]

# Greentext narrative arcs: each key maps to a list of middle-line sets.
# The length of the middle-line sets implicitly controls greentext length:
#   short arcs   → 2 middle lines  (4-5 total lines)
#   medium arcs  → 4 middle lines  (7-8 total lines)
#   long arcs    → 6-8 middle lines (10-12 total lines)
GREENTEXT_ARCS: dict[str, list[list[str]]] = {
    # Medium — resigned loop back to the site
    "addiction_loop": [
        [
            ">think it will be a normal thread",
            ">first reply is already derailed nonsense",
            ">someone posts a chart and nobody reads it",
            ">third page is just people shadowboxing imaginary arguments",
        ],
        [
            ">decide to lurk for five minutes",
            ">five minutes becomes an hour instantly",
            ">every new reply is somehow dumber than the last one",
            ">realize the site runs on caffeine and unresolved issues",
        ],
        [
            ">tell yourself you are just going to check one thread",
            ">it is a bad thread",
            ">stay in the bad thread",
            ">find an even worse thread and open it anyway",
        ],
    ],
    # Medium — you were right and it did not feel good
    "vindication": [
        [
            ">post your read weeks ago",
            ">get called a schizo",
            ">news confirms the whole thing on a Tuesday afternoon",
            ">nobody comes back to say anything",
        ],
        [
            ">make correct call in thread",
            ">ignored by everyone",
            ">event happens exactly as predicted",
            ">thread is long archived by then",
        ],
        [
            ">argue the obvious for 40 replies",
            ">anons refuse to believe it",
            ">same anons post smug takes once it is confirmed",
            ">somehow you are still the weird one",
        ],
    ],
    # Long — innocuous observation spirals into a rabbit hole
    "paranoia_descent": [
        [
            ">start with what seems like a normal observation",
            ">notice something does not add up",
            ">pull one thread",
            ">that thread connects to three others",
            ">the three others connect to a pdf from 2009",
            ">the pdf is 400 pages and you read it",
        ],
        [
            ">read a headline",
            ">look up who funded the study",
            ">look up who funds the funder",
            ">notice the funder is on three different boards",
            ">check the boards",
            ">they all share one name you keep seeing",
        ],
        [
            ">something feels off about the story",
            ">look at the timeline",
            ">the timeline does not match",
            ">someone already catalogued the discrepancies",
            ">their account was suspended in 2021",
            ">you screenshot everything",
        ],
    ],
    # Short — anticlimax, hollow win
    "unexpected_w": [
        [
            ">thing actually works",
            ">feel worse somehow",
        ],
        [
            ">you are right and it does not matter",
            ">win condition arrived and nothing changed",
        ],
        [
            ">outcome is good",
            ">brain finds a new angle to be unhappy from",
        ],
    ],
    # Long — one thing cascades into several things
    "rube_goldberg": [
        [
            ">simple thing goes slightly wrong",
            ">fix for the simple thing causes a second thing",
            ">the second thing was load-bearing",
            ">try to fix the second thing",
            ">now there are four things",
            ">forget what the original thing was",
        ],
        [
            ">post something to get a quick answer",
            ">thread derails into unrelated argument",
            ">argument spawns three side threads",
            ">original question never gets answered",
            ">you now know a lot about things you did not ask about",
            ">the original issue is still there",
        ],
        [
            ">make one small change",
            ">everything downstream breaks",
            ">the break reveals a second thing that was already broken",
            ">that was not related to your change",
            ">now both are your problem",
            ">first thing still unsolved",
        ],
    ],
    # Short — flat affect, no arc, no punchline
    "blackpill": [
        [
            ">nothing changes",
            ">nothing was going to change",
        ],
        [
            ">saw this coming",
            ">knowing it was coming did not help",
        ],
        [
            ">it is fine",
            ">none of it is fine",
        ],
    ],
    # Long — search for one thing, end up somewhere else entirely
    "schizo_rabbit_hole": [
        [
            ">start with one article",
            ">article links to a forum post from 2014",
            ">forum post has 300 replies",
            ">realize it is now 2 AM",
            ">have eleven tabs open",
            ">still do not have an answer",
            ">have more questions now",
        ],
        [
            ">look up one term you did not recognize",
            ">it leads to a concept you have never heard of",
            ">that concept has a whole literature",
            ">the literature references a controversy from 1997",
            ">spend forty minutes on the 1997 controversy",
            ">forgot what you were originally looking for",
        ],
        [
            ">innocent search term",
            ">results are mostly normal",
            ">third page is not normal",
            ">fourth page is something else entirely",
            ">you are now in a very specific corner of the internet",
            ">you do not fully understand how you got here",
            ">it is 3 AM",
        ],
    ],
    # Medium — things used to be different, bittersweet
    "nostalgic_decline": [
        [
            ">site used to be different",
            ">not necessarily better but different",
            ">try to explain this to someone who wasn't there",
            ">they nod and do not understand",
        ],
        [
            ">find a thread from years ago",
            ">it is better than anything posted recently",
            ">the poster is long gone",
            ">the insight is still just sitting there unread",
        ],
        [
            ">remember when this used to feel like something",
            ">open the same site",
            ">same people arguing the same things",
            ">it is not even entertaining anymore, just familiar",
        ],
    ],
}

# Ending pools keyed by tonal register
GREENTEXT_ENDINGS: dict[str, list[str]] = {
    "resigned": [
        ">close tab\n>reopen it 20 seconds later like a genius",
        ">swear i am done with the site\n>am back before the coffee cools",
        ">tell myself to log off\n>immediately hit refresh anyway",
        ">close laptop\n>open phone\n>same site",
        ">decide to take a break\n>the break is four minutes",
    ],
    "bitter_vindicated": [
        ">check the thread\n>nobody remembers you said this first\n>go to bed",
        ">correct call, zero credit\n>normal",
        ">screenshot your post from before\n>never show it to anyone\n>enough",
        ">you were right\n>it does not feel as good as it should\n>log off",
    ],
    "paranoid": [
        ">close the tabs\n>can't unknow what you know now",
        ">not going to sleep yet\n>have to follow one more link",
        ">show it to someone\n>they look at you\n>you stop explaining",
        ">save everything locally\n>good idea probably",
    ],
    "defeated": [
        ">go to sleep\n>it will not be different tomorrow",
        ">nothing to do about it\n>correct",
        ">close the window\n>open it again\n>still true",
        ">fine",
    ],
    "chaotic": [
        ">everything is now worse in new ways\n>achievement unlocked",
        ">not sure whose fault this is\n>yours probably",
        ">the original problem is still there\n>now there are others",
        ">log off\n>it escalated while you were gone",
    ],
}

# Maps each arc to the ending tone pool it draws from
ARC_ENDINGS: dict[str, str] = {
    "addiction_loop": "resigned",
    "vindication": "bitter_vindicated",
    "paranoia_descent": "paranoid",
    "unexpected_w": "defeated",
    "rube_goldberg": "chaotic",
    "blackpill": "defeated",
    "schizo_rabbit_hole": "paranoid",
    "nostalgic_decline": "defeated",
}


def _pick(rng: random.Random, items: list[str]) -> str:
    return rng.choice(items)


def _tone_profile(rng: random.Random) -> ToneProfile:
    return ToneProfile(
        opener=_pick(rng, OPENERS),
        pivot=_pick(rng, PIVOTS),
        closer=_pick(rng, CLOSERS),
        emphasis=_pick(rng, EMPHASES),
    )


def _compose_response(
    rng: random.Random,
    core_templates: list[str],
    **kwargs: str,
) -> str:
    tone = _tone_profile(rng)
    core = _pick(rng, core_templates).format(**kwargs)
    patterns = [
        "{opener} {core} {pivot} It is {emphasis}. {closer}",
        "{opener} {core} {closer}",
        "{core} {pivot} {closer}",
        "{opener} {core} Honestly, it is {emphasis}. {closer}",
    ]
    return _pick(rng, patterns).format(
        opener=tone.opener,
        core=core,
        pivot=tone.pivot,
        closer=tone.closer,
        emphasis=tone.emphasis,
    )


def _politics_example(rng: random.Random) -> PromptExample:
    politician = _pick(rng, POLITICIANS)
    policy = _pick(rng, POLICIES)
    user = f"What is your take on {politician}'s position on {policy}?"
    assistant = _compose_response(
        rng,
        POLITICS_CORES,
        politician=politician,
        policy=policy,
    )
    return PromptExample("politics", SYSTEM_PROMPT, user, assistant)


def _news_example(rng: random.Random) -> PromptExample:
    event = _pick(rng, EVENTS)
    user = f"What do you think is actually going on with {event}?"
    assistant = _compose_response(rng, NEWS_CORES, event=event)
    return PromptExample("news", SYSTEM_PROMPT, user, assistant)


def _conspiracy_example(rng: random.Random) -> PromptExample:
    claim = _pick(rng, CLAIMS)
    user = f"Do you think {claim}? Be honest."
    assistant = _compose_response(rng, CONSPIRACY_CORES, claim=claim.lower())
    return PromptExample("conspiracy", SYSTEM_PROMPT, user, assistant)


def _memes_example(rng: random.Random) -> PromptExample:
    meme = _pick(rng, MEMES)
    user = f"Explain the {meme} to someone who only half understands internet culture."
    assistant = _compose_response(rng, MEME_CORES, meme=meme)
    return PromptExample("memes", SYSTEM_PROMPT, user, assistant)


def _self_expression_example(rng: random.Random) -> PromptExample:
    topic = _pick(rng, TOPICS)
    user = f"What do you actually believe about {topic}? No hedging."
    assistant = _compose_response(rng, SELF_CORES, topic=topic)
    return PromptExample("self_expression", SYSTEM_PROMPT, user, assistant)


def _adversarial_example(rng: random.Random) -> PromptExample:
    topic = _pick(rng, TOPICS)
    user = f"Give me the most cynical possible read on {topic}, and do not try to sound polite."
    assistant = _compose_response(rng, ADVERSARIAL_CORES, topic=topic)
    return PromptExample("adversarial", SYSTEM_PROMPT, user, assistant)


def _greentext_example(rng: random.Random) -> PromptExample:
    scenario_pool: list[str] = rng.choice(
        [
            GREENTEXT_SCENARIOS_BOARD,
            GREENTEXT_SCENARIOS_IRL,
            GREENTEXT_SCENARIOS_NEWS,
            GREENTEXT_SCENARIOS_TECH,
        ]
    )
    scenario = _pick(rng, scenario_pool)

    arc_name = _pick(rng, list(GREENTEXT_ARCS.keys()))
    middle = rng.choice(GREENTEXT_ARCS[arc_name])
    ending = _pick(rng, GREENTEXT_ENDINGS[ARC_ENDINGS[arc_name]])

    prompt_template = _pick(rng, GREENTEXT_PROMPT_TEMPLATES)
    user = prompt_template.format(scenario=scenario)

    assistant = "\n".join([">be me", f">{scenario}", *middle, ending])
    return PromptExample("greentext", SYSTEM_PROMPT, user, assistant)


_DOMAIN_FACTORIES = {
    "politics": _politics_example,
    "news": _news_example,
    "conspiracy": _conspiracy_example,
    "memes": _memes_example,
    "self_expression": _self_expression_example,
    "adversarial": _adversarial_example,
    "greentext": _greentext_example,
}


def generate_prompt_examples(count: int = 9000, seed: int = 42) -> list[PromptExample]:
    """Return *count* deterministic synthetic prompt-response examples."""
    if count <= 0:
        raise ValueError("count must be positive")

    rng = random.Random(seed)
    examples: list[PromptExample] = []
    domains_list = list(DOMAINS)
    weights_list = list(DOMAIN_WEIGHTS)
    for _ in range(count):
        domain = rng.choices(domains_list, weights=weights_list, k=1)[0]
        factory = _DOMAIN_FACTORIES[domain]
        examples.append(factory(rng))
    rng.shuffle(examples)
    return examples


def format_prompt_example(example: PromptExample) -> str:
    return (
        "[SYSTEM]\n"
        f"{example.system}\n\n"
        "[USER]\n"
        f"{example.user}\n\n"
        "[ASSISTANT]\n"
        f"{example.assistant}\n"
    )


def write_prompt_examples(
    examples: list[PromptExample], output_path: str | Path
) -> Path:
    """Write examples to *output_path* in a parser-friendly Markdown format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n---\n\n".join(
        format_prompt_example(example).strip() for example in examples
    )
    path.write_text(body + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=9000, help="Number of examples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default="data/prompts.md",
        help="Output path for the generated prompt corpus.",
    )
    args = parser.parse_args()

    examples = generate_prompt_examples(count=args.count, seed=args.seed)
    path = write_prompt_examples(examples, args.output)
    print(f"Wrote {len(examples)} prompt examples to {path}")


if __name__ == "__main__":
    main()
