"""
proposer.py
-----------
The hypothesis generator -- the part that makes this an autonomous *research*
agent rather than a hyperparameter sweep.

Each round the proposer reads the run history (what was tried, what it scored,
what crashed), plus the organizers' measured dead ends and open directions, and
returns a new `Candidate`: a stated hypothesis plus the code and/or parameter
changes to test it.

Backends, selected automatically:
  OpenAIProposer  -- LLM-driven, used when OPENAI_API_KEY is set
  StaticProposer  -- walks a fixed pool; the offline fallback so the pipeline
                     still runs (and stays reproducible) with no API key
"""

from __future__ import annotations

import json
import os
import textwrap

from candidate import Candidate
import scoring

PROPOSER_MODEL = os.getenv("PROPOSER_MODEL", "gpt-5.1")

SYSTEM_PROMPT = """\
You are an autonomous machine-learning research agent competing on the \
KuaiRand-Pure recommendation benchmark. Each round you propose ONE concrete, \
testable change to the pipeline, and you will be shown what it scored.

TASK
  Within-user ranking over logged impressions. For each user you must order the \
videos they were shown so the ones they actually long-viewed come first.
  Label   : long_view (0/1)
  Metrics : GAUC and nDCG@5; primary = mean of the two
  Baseline to beat (validation): primary 0.6016
  Oracle ceiling (validation)  : primary 0.8484 -- 27% of users have no positive \
label at all, so 1.0 is unreachable. Judge progress against the ceiling.

MODEL
  LightGBM. You may set any LightGBM parameter via param_overrides, including \
"objective": "lambdarank" (listwise ranking, grouped per user -- the harness \
handles the grouping for you) or "rank_xendcg". The default is "binary" \
(pointwise logloss).

CODE CONTRACT
  If your hypothesis needs new features, return Python source in `code` that \
defines exactly two functions:

    def fit(train_df):
        # Called on the TRAIN split ONLY. Compute any statistics here.
        # Returns any object; it is passed unchanged to apply().
        return state

    def apply(df, state):
        # Called on train, validation and test alike.
        # Add one or more new columns and return df.
        return df

  Rules:
  - pandas is available as `pd`, numpy as `np`. Do not import anything else.
  - NEVER compute statistics from `df` inside apply() -- that leaks validation
    and test information. All statistics must come from fit() via `state`.
  - apply() MUST add at least one new column, and the same columns on every split.
  - Handle unseen keys (a video_id in test that is not in train) with a sensible
    fallback, e.g. a global prior.
  - Leave `code` as an empty string if you are only changing parameters.

STRATEGY
  Do not repeat anything listed under ALREADY TRIED or MEASURED DEAD ENDS. \
Prefer changes that attack the current bottleneck. Read the score history: if \
the last few changes moved the score by less than 0.002, the current line of \
attack is exhausted -- change *kind* of idea, not magnitude.
  State your reasoning in `hypothesis` in one or two sentences: what you expect \
to happen and why. That field is graded.
"""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "candidate",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "short snake_case identifier for this experiment",
                },
                "hypothesis": {
                    "type": "string",
                    "description": "what you expect to happen and why, 1-2 sentences",
                },
                "code": {
                    "type": "string",
                    "description": "Python source defining fit(train_df) and "
                                   "apply(df, state); empty string if none",
                },
                "param_overrides_json": {
                    "type": "string",
                    "description": "JSON object of LightGBM parameter overrides, "
                                   "e.g. {\"objective\": \"lambdarank\"}; \"{}\" if none",
                },
            },
            "required": ["name", "hypothesis", "code", "param_overrides_json"],
            "additionalProperties": False,
        },
    },
}


def _history_block(history: list[dict], limit: int = 12) -> str:
    if not history:
        return "  (nothing tried yet -- this is iteration 0)"
    lines = []
    for r in history[-limit:]:
        if r.get("status") == "success":
            m = r["metrics"]
            d = r["delta_vs_official_baseline"]["primary_delta"]
            lines.append(
                f"  [{r['iteration']}] {r['candidate_name']}: "
                f"primary {m['primary']:.4f} (delta vs baseline {d:+.4f})\n"
                f"      hypothesis: {r['hypothesis']}"
            )
        else:
            lines.append(
                f"  [{r['iteration']}] {r['candidate_name']}: FAILED -- "
                f"{str(r.get('error'))[:200]}\n"
                f"      hypothesis: {r['hypothesis']}"
            )
    return "\n".join(lines)


def build_context(history: list[dict], columns: list[str]) -> str:
    from agent_loop import KNOWN_DEAD_ENDS, OPEN_DIRECTIONS

    return textwrap.dedent(f"""\
        SCORE HISTORY (validation)
        {_history_block(history)}

        MEASURED DEAD ENDS (the organizers ran these; do not repeat them)
        {chr(10).join('  - ' + d for d in KNOWN_DEAD_ENDS)}

        UNEXPLORED DIRECTIONS (organizer-ranked, most promising first)
        {chr(10).join(f'  {i+1}. {d}' for i, d in enumerate(OPEN_DIRECTIONS))}

        COLUMNS AVAILABLE in the DataFrame passed to fit/apply
        {', '.join(columns)}

        Propose iteration {len(history)}.
        """)


class StaticProposer:
    """Offline fallback: walk a fixed pool. Deterministic, no API key needed."""

    source = "static"

    def __init__(self, pool: list[Candidate]):
        self.pool = pool
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def propose(self, history, columns, i):
        return self.pool[i] if i < len(self.pool) else None


class OpenAIProposer:
    """LLM-driven proposer. Reads the run log, writes the next experiment."""

    source = "llm"

    def __init__(self, model: str = PROPOSER_MODEL, seed_pool: list[Candidate] | None = None):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.seed_pool = seed_pool or []
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def propose(self, history, columns, i):
        # Seed the first iteration with a known-good baseline so there is
        # always a scored reference point before the LLM starts exploring.
        if i == 0 and self.seed_pool:
            return self.seed_pool[0]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_context(history, columns)},
            ],
            response_format=RESPONSE_SCHEMA,
        )

        if resp.usage:
            self.usage["input_tokens"] += resp.usage.prompt_tokens or 0
            self.usage["output_tokens"] += resp.usage.completion_tokens or 0

        payload = json.loads(resp.choices[0].message.content)

        try:
            overrides = json.loads(payload.get("param_overrides_json") or "{}")
            if not isinstance(overrides, dict):
                overrides = {}
        except json.JSONDecodeError:
            overrides = {}

        code = (payload.get("code") or "").strip() or None

        return Candidate(
            name=payload["name"],
            hypothesis=payload["hypothesis"],
            feature_pipeline=[],          # agent code supplies the features
            param_overrides=overrides,
            code=code,
            source="llm",
        )


def make_proposer(seed_pool: list[Candidate] | None = None, force_static: bool = False):
    """Pick a backend: LLM when a key is available, static pool otherwise."""
    if not force_static and os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIProposer(seed_pool=seed_pool)
        except Exception as exc:  # SDK missing, bad key, etc.
            print(f"  [proposer] OpenAI unavailable ({exc}); falling back to static pool")
    return StaticProposer(seed_pool or [])
