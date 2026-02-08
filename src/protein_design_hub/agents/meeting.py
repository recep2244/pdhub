"""Meeting runner for LLM agent discussions.

Supports two meeting types modelled after the Virtual-Lab pattern:
  * **team**:       team lead + N members discuss an agenda in rounds.
  * **individual**: one agent + scientific critic iterate.

Each meeting produces a discussion transcript (list of dicts) and
optionally returns the final summary.  Transcripts are saved to disk
as JSON and Markdown.

The LLM backend is read from ``Settings.llm`` so it works out of the
box with **Ollama** on localhost (no API key needed).

Reference: https://github.com/zou-group/virtual-lab
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

from protein_design_hub.agents.llm_agent import LLMAgent
from protein_design_hub.agents import prompts as P

# Re-export for convenience
Discussion = List[Dict[str, str]]


# ── LLM client helper ───────────────────────────────────────────────

def _get_llm_client():
    """Build an OpenAI client using Settings.llm (works with Ollama too)."""
    from openai import OpenAI
    from protein_design_hub.core.config import get_settings

    cfg = get_settings().llm.resolve()
    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key), cfg


def _call_llm(
    agent: LLMAgent,
    messages: List[Dict[str, str]],
    temperature: float | None = None,
) -> str:
    """Call the LLM for *agent* using the configured backend.

    Works with Ollama, OpenAI, vLLM, LM Studio, or any OpenAI-compatible
    server.
    """
    client, cfg = _get_llm_client()
    model = agent.resolved_model
    temp = temperature if temperature is not None else cfg.temperature
    agent_messages = [agent.system_message] + messages

    kwargs: dict = dict(
        model=model,
        messages=agent_messages,  # type: ignore[arg-type]
        temperature=temp,
    )
    if cfg.max_tokens is not None:
        kwargs["max_tokens"] = cfg.max_tokens

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


# ── saving helpers ──────────────────────────────────────────────────

def _save_discussion(
    save_dir: Path,
    save_name: str,
    discussion: Discussion,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{save_name}.json", "w") as f:
        json.dump(discussion, f, indent=4)
    with open(save_dir / f"{save_name}.md", "w", encoding="utf-8") as f:
        for turn in discussion:
            f.write(f"## {turn['agent']}\n\n{turn['message']}\n\n")


def get_summary(discussion: Discussion) -> str:
    """Return the last agent message (meeting summary)."""
    return discussion[-1]["message"]


# ── main entry point ────────────────────────────────────────────────

def run_meeting(
    meeting_type: Literal["team", "individual"],
    agenda: str,
    save_dir: Path,
    save_name: str = "discussion",
    # team meeting
    team_lead: Optional[LLMAgent] = None,
    team_members: Optional[Sequence[LLMAgent]] = None,
    # individual meeting
    team_member: Optional[LLMAgent] = None,
    critic: Optional[LLMAgent] = None,
    # options
    agenda_questions: Sequence[str] = (),
    agenda_rules: Sequence[str] = (),
    summaries: Sequence[str] = (),
    contexts: Sequence[str] = (),
    num_rounds: int = 1,
    temperature: float | None = None,
    return_summary: bool = False,
) -> Optional[str]:
    """Run a meeting with LLM agents.

    Args:
        meeting_type: ``"team"`` or ``"individual"``.
        agenda: Free-text agenda string.
        save_dir: Directory in which to persist JSON + Markdown transcripts.
        save_name: Base filename for the saved transcript.
        team_lead: Team lead agent (team meetings only).
        team_members: Sequence of member agents (team meetings only).
        team_member: The single agent for individual meetings.
        critic: The critic agent for individual meetings (defaults to
            ``SCIENTIFIC_CRITIC`` from :pymod:`scientists`).
        agenda_questions: Questions the meeting must answer.
        agenda_rules: Rules that must be followed.
        summaries: Summaries from prior meetings for context.
        contexts: Additional context strings.
        num_rounds: Number of discussion rounds.
        temperature: LLM sampling temperature (None = use config default).
        return_summary: If *True*, return the summary string.

    Returns:
        Meeting summary if *return_summary* is True, else None.
    """
    # ── validation ──────────────────────────────────────────────────
    if meeting_type == "team":
        if team_lead is None or team_members is None or len(team_members) == 0:
            raise ValueError("Team meeting requires team_lead and team_members")
        if team_member is not None:
            raise ValueError("team_member is not used in team meetings")
        if team_lead in team_members:
            raise ValueError("team_lead must not be in team_members")
    elif meeting_type == "individual":
        if team_member is None:
            raise ValueError("Individual meeting requires team_member")
        if team_lead is not None or team_members is not None:
            raise ValueError("team_lead/team_members are not used in individual meetings")
        if critic is None:
            from protein_design_hub.agents.scientists import SCIENTIFIC_CRITIC
            critic = SCIENTIFIC_CRITIC
    else:
        raise ValueError(f"Invalid meeting type: {meeting_type}")

    start_time = time.time()

    discussion: Discussion = []
    messages: List[Dict[str, str]] = []

    # ── TEAM meeting ────────────────────────────────────────────────
    if meeting_type == "team":
        assert team_lead is not None and team_members is not None
        team: list[LLMAgent] = [team_lead] + list(team_members)

        initial = P.team_meeting_start(
            team_lead=team_lead,
            team_members=team_members,
            agenda=agenda,
            questions=agenda_questions,
            rules=agenda_rules,
            summaries=summaries,
            contexts=contexts,
            num_rounds=num_rounds,
        )
        messages.append({"role": "user", "content": initial})
        discussion.append({"agent": "User", "message": initial})

        for round_idx in range(num_rounds + 1):
            round_num = round_idx + 1
            for agent in team:
                # choose prompt
                if agent == team_lead:
                    if round_idx == 0:
                        prompt = P.team_lead_initial(team_lead)
                    elif round_idx == num_rounds:
                        prompt = P.team_lead_final(
                            team_lead, agenda, agenda_questions, agenda_rules,
                        )
                    else:
                        prompt = P.team_lead_intermediate(
                            team_lead, round_num - 1, num_rounds,
                        )
                else:
                    prompt = P.team_member_prompt(agent, round_num, num_rounds)

                messages.append({"role": "user", "content": prompt})
                discussion.append({"agent": "User", "message": prompt})

                reply = _call_llm(agent, messages, temperature)
                messages.append({"role": "assistant", "content": reply})
                discussion.append({"agent": agent.title, "message": reply})

                # In the final round only the lead speaks
                if round_idx == num_rounds:
                    break

    # ── INDIVIDUAL meeting ──────────────────────────────────────────
    else:
        assert team_member is not None and critic is not None
        team = [team_member, critic]

        for round_idx in range(num_rounds + 1):
            for agent in team:
                if agent == critic:
                    prompt = P.critic_prompt(critic, team_member)
                else:
                    if round_idx == 0:
                        prompt = P.individual_start(
                            team_member, agenda, agenda_questions,
                            agenda_rules, summaries, contexts,
                        )
                    else:
                        prompt = P.agent_revision_prompt(critic, team_member)

                messages.append({"role": "user", "content": prompt})
                discussion.append({"agent": "User", "message": prompt})

                reply = _call_llm(agent, messages, temperature)
                messages.append({"role": "assistant", "content": reply})
                discussion.append({"agent": agent.title, "message": reply})

                if round_idx == num_rounds:
                    break

    elapsed = time.time() - start_time
    print(f"Meeting completed in {int(elapsed // 60)}m {int(elapsed % 60):02d}s")

    _save_discussion(save_dir, save_name, discussion)

    if return_summary:
        return get_summary(discussion)
    return None
