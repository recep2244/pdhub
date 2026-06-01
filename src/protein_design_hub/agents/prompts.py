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
        "### Team Member Input\n"
        "Summarize the key point raised by **each team member by name**. "
        "Note where members agreed, disagreed, or built on each other's insights.",
        "### Key Debates\n"
        "Identify the most important scientific disagreements or tensions that arose. "
        "For each: state the opposing views, who held them, and how the debate was resolved "
        "(or why it remains open). If no genuine disagreements arose, note the strongest "
        "caveat or uncertainty raised.",
        "### Cross-disciplinary Insights\n"
        "Highlight observations where two or more disciplines shed light on the same "
        "issue from different angles. These are the most valuable outputs — identify "
        "where structural, biophysical, and ML perspectives converge or diverge.",
        "### Recommendation\n"
        "Provide a clear, specific, and actionable recommendation. Justify it "
        "with concrete metrics and thresholds (e.g. 'proceed if pLDDT > 80', "
        "'refine if clash score > 20'). Avoid vague statements. Cite specific "
        "numbers from the data.",
    ]
    if has_questions:
        parts.append(
            "### Answers\nFor each agenda question provide:\n"
            "Answer: ... (include specific numbers, thresholds, or tool names)\n"
            "Justification: ... (cite evidence from the data or literature)\n"
            "Dissenting view: ... (if any team member disagreed, note it here)"
        )
    parts.append(
        "### Next Steps\n"
        "Outline concrete next steps with specific actions "
        "(which tools to run, which parameters to use, which metrics to check). "
        "Assign each step to a responsible domain expert."
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
        f"Before stating your own view, **explicitly acknowledge at least one point "
        f"made by a previous speaker** — either agreeing with specific evidence, "
        f"qualifying their claim with your domain expertise, or respectfully "
        f"challenging it with a counter-argument. Then add your domain-specific "
        f"perspective that has NOT yet been covered. "
        f'If you have nothing to add or challenge, you may say "pass". '
        f"Scientific disagreement is welcome — be direct, not vague."
    )


def team_lead_intermediate(team_lead: LLMAgent, round_num: int, num_rounds: int) -> str:
    return (
        f"This concludes round {round_num} of {num_rounds}. "
        f"{team_lead}, please {SYNTHESISE}. "
        f"Specifically: (1) call out the most important **agreement** — where multiple "
        f"disciplines independently reached the same conclusion; (2) call out the most "
        f"important **disagreement** — where team members diverged, and ask a probing "
        f"follow-up question to resolve it in the next round; (3) identify any "
        f"**critical gap** — a domain perspective that has not yet been represented."
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
        "Structure your critique around three elements: "
        "(1) **What is correct and well-supported** — acknowledge strong points with "
        "specific references to the data or thresholds cited; "
        "(2) **What is missing or insufficiently justified** — identify claims that lack "
        "quantitative support, skipped failure modes, or overlooked caveats; "
        "(3) **What is potentially wrong** — challenge any assertion where the evidence "
        "is weak or contradicts established thresholds, naming the specific value or "
        "claim you are questioning. "
        "Prioritize simple solutions over complex ones but demand more detail where "
        "detail is lacking. Validate whether the answer adheres to the agenda and rules."
    )


def agent_revision_prompt(critic: LLMAgent, agent: LLMAgent) -> str:
    return (
        f"{agent}, please revise your answer to address {critic}'s "
        "most recent feedback. Your goal is to better address the agenda."
    )


# ── cross-examination prompt ────────────────────────────────────────

def cross_examination_prompt(critic: LLMAgent, target_member: LLMAgent) -> str:
    """Targeted challenge from the critic to a specific team member."""
    return (
        f"{critic}, please directly challenge {target_member}'s most recent contribution. "
        f"Identify the single most important claim they made and: "
        f"(1) state the claim precisely, quoting specific numbers if possible; "
        f"(2) provide the strongest counter-argument or alternative interpretation; "
        f"(3) ask {target_member} one sharp question that would resolve the disagreement. "
        f"Be specific and quantitative — vague disagreements are not useful."
    )


def member_rebuttal_prompt(member: LLMAgent, critic: LLMAgent) -> str:
    """Member defends or concedes in response to critic challenge."""
    return (
        f"{member}, please respond to {critic}'s challenge. "
        f"Either (a) defend your position with additional evidence or reasoning, "
        f"or (b) concede the point and revise your recommendation accordingly. "
        f"Do not simply restate your earlier view — engage directly with the "
        f"counter-argument raised. If you concede, specify exactly what changes."
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
