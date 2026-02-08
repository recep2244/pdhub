"""Prompt templates for LLM agent meetings.

Follows the Virtual-Lab meeting pattern adapted for protein design:
  * team meetings   – team lead + members discuss an agenda in rounds
  * individual meetings – one agent + scientific critic iterate

Reference: https://github.com/zou-group/virtual-lab
"""

from __future__ import annotations

from typing import Iterable, Sequence

from protein_design_hub.agents.llm_agent import LLMAgent


# ── helper formatters ───────────────────────────────────────────────

def _numbered_list(items: Iterable[str]) -> str:
    return "\n\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))


def format_agenda(agenda: str, intro: str = "Here is the agenda for the meeting:") -> str:
    return f"{intro}\n\n{agenda}\n\n"


def format_questions(
    questions: Sequence[str],
    intro: str = "Here are the agenda questions that must be answered:",
) -> str:
    return f"{intro}\n\n{_numbered_list(questions)}\n\n" if questions else ""


def format_rules(
    rules: Sequence[str],
    intro: str = "Here are the agenda rules that must be followed:",
) -> str:
    return f"{intro}\n\n{_numbered_list(rules)}\n\n" if rules else ""


def format_references(
    references: Sequence[str],
    ref_type: str,
    intro: str,
) -> str:
    if not references:
        return ""
    blocks = [
        f"[begin {ref_type} {i + 1}]\n\n{ref}\n\n[end {ref_type} {i + 1}]"
        for i, ref in enumerate(references)
    ]
    return f"{intro}\n\n" + "\n\n".join(blocks) + "\n\n"


# ── summary structure ───────────────────────────────────────────────

def summary_structure(has_questions: bool) -> str:
    parts = [
        "### Agenda\nRestate the agenda in your own words.",
        "### Team Member Input\nSummarize all important points raised by each team member.",
        "### Recommendation\n"
        "Provide a clear, specific, and actionable recommendation. Justify it "
        "with concrete metrics and thresholds (e.g. 'proceed if pLDDT > 80', "
        "'refine if clash score > 20'). Avoid vague statements.",
    ]
    if has_questions:
        parts.append(
            "### Answers\nFor each agenda question provide:\n"
            "Answer: ... (include specific numbers, thresholds, or tool names)\n"
            "Justification: ... (cite evidence from the data or literature)"
        )
    parts.append(
        "### Next Steps\n"
        "Outline concrete next steps with specific actions "
        "(which tools to run, which parameters to use, which metrics to check)."
    )
    return "\n\n".join(parts)


# ── team meeting prompts ────────────────────────────────────────────

SYNTHESISE = (
    "synthesize the points raised by each team member, make decisions "
    "regarding the agenda based on team member input, and ask follow-up "
    "questions to gather more information"
)

SUMMARISE = (
    "summarize the meeting in detail for future discussions, provide a "
    "specific recommendation regarding the agenda, and answer the agenda "
    "questions (if any) based on the discussion while strictly adhering "
    "to the agenda rules (if any)"
)


def team_meeting_start(
    team_lead: LLMAgent,
    team_members: Sequence[LLMAgent],
    agenda: str,
    questions: Sequence[str] = (),
    rules: Sequence[str] = (),
    summaries: Sequence[str] = (),
    contexts: Sequence[str] = (),
    num_rounds: int = 1,
) -> str:
    member_names = ", ".join(str(m) for m in team_members)
    return (
        f"This is the beginning of a team meeting to discuss your research project. "
        f"This is a meeting with the team lead, {team_lead}, and the following team members: "
        f"{member_names}.\n\n"
        f"{format_references(contexts, 'context', 'Here is context for this meeting:')}"
        f"{format_references(summaries, 'summary', 'Here are summaries of previous meetings:')}"
        f"{format_agenda(agenda)}"
        f"{format_questions(questions)}"
        f"{format_rules(rules)}"
        f"{team_lead} will convene the meeting. "
        f"Then, each team member will provide their thoughts one-by-one. "
        f"After all team members have spoken, {team_lead} will {SYNTHESISE}. "
        f"This will continue for {num_rounds} round(s). "
        f"Once done, {team_lead} will {SUMMARISE}."
    )


def team_lead_initial(team_lead: LLMAgent) -> str:
    return (
        f"{team_lead}, please provide your initial thoughts on the agenda "
        f"as well as any questions to guide the team."
    )


def team_member_prompt(member: LLMAgent, round_num: int, num_rounds: int) -> str:
    return (
        f"{member}, please provide your thoughts (round {round_num} of {num_rounds}). "
        f'If you have nothing new to add, you may say "pass". '
        f"You can and should politely disagree if you have a different perspective."
    )


def team_lead_intermediate(team_lead: LLMAgent, round_num: int, num_rounds: int) -> str:
    return (
        f"This concludes round {round_num} of {num_rounds}. "
        f"{team_lead}, please {SYNTHESISE}."
    )


def team_lead_final(
    team_lead: LLMAgent,
    agenda: str,
    questions: Sequence[str] = (),
    rules: Sequence[str] = (),
) -> str:
    return (
        f"{team_lead}, please {SUMMARISE}.\n\n"
        f"{format_agenda(agenda, intro='As a reminder, here is the agenda:')}"
        f"{format_questions(questions, intro='Reminder – agenda questions:')}"
        f"{format_rules(rules, intro='Reminder – agenda rules:')}"
        f"Your summary should follow this structure:\n\n"
        f"{summary_structure(has_questions=len(questions) > 0)}"
    )


# ── individual meeting prompts ──────────────────────────────────────

def individual_start(
    agent: LLMAgent,
    agenda: str,
    questions: Sequence[str] = (),
    rules: Sequence[str] = (),
    summaries: Sequence[str] = (),
    contexts: Sequence[str] = (),
) -> str:
    return (
        f"This is the beginning of an individual meeting with {agent} "
        f"to discuss your research project.\n\n"
        f"{format_references(contexts, 'context', 'Here is context:')}"
        f"{format_references(summaries, 'summary', 'Here are summaries of previous meetings:')}"
        f"{format_agenda(agenda)}"
        f"{format_questions(questions)}"
        f"{format_rules(rules)}"
        f"{agent}, please provide your response to the agenda."
    )


def critic_prompt(critic: LLMAgent, agent: LLMAgent) -> str:
    return (
        f"{critic}, please critique {agent}'s most recent answer. "
        "Suggest improvements that directly address the agenda. "
        "Prioritize simple solutions over complex ones but demand more "
        "detail where detail is lacking. "
        "Validate whether the answer adheres to the agenda and rules."
    )


def agent_revision_prompt(critic: LLMAgent, agent: LLMAgent) -> str:
    return (
        f"{agent}, please revise your answer to address {critic}'s "
        "most recent feedback. Your goal is to better address the agenda."
    )


# ── merge prompt ────────────────────────────────────────────────────

MERGE_PROMPT = (
    "Please read the summaries of multiple separate meetings about the same "
    "agenda. Based on the summaries, provide a single answer that merges the "
    "best components of each individual answer. Explain what components came "
    "from each meeting and why."
)


# ── protein-design-specific coding rules ────────────────────────────

CODING_RULES = (
    "Your code must be self-contained with appropriate imports.",
    "Your code must not include undefined variables or functions.",
    "Your code must not include pseudocode; it must be fully functional.",
    "Your code must not include hard-coded examples.",
    "If user-provided values are needed, parse them from the command line.",
    "Code must be well-documented with docstrings, comments, and type hints.",
)
