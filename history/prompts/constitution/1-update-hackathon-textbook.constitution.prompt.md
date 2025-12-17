---
id: 1
title: update-hackathon-textbook
stage: constitution
date: 2025-12-08
surface: agent
model: Qwen
feature: null
branch: main
user: user
command: /sp.constitution
labels: ["constitution", "hackathon", "textbook", "ai-native"]
links: {spec: null, ticket: null, adr: null, pr: null}
files_yaml: [
  - ".specify/memory/constitution.md",
  - "history/prompts/constitution/1-update-hackathon-textbook.constitution.prompt.md"
]
tests_yaml: []
---

# Constitution Update: AI-Native Physical & Humanoid Robotics Textbook

## Outcome
Updated the project constitution file to reflect the requirements for the Hackathon I: AI-Native Physical & Humanoid Robotics Textbook project. The constitution now includes specific principles for safe, ethical, and efficient operations while supporting RAG chatbots, personalized content, and multi-language translation.

## Evaluation
The constitution file was successfully updated with 8 core principles, permissions, restrictions, and governance guidelines specific to the AI-Native Physical & Humanoid Robotics Textbook project. A Sync Impact Report was included as an HTML comment at the top of the file. All template placeholders were properly filled with the appropriate content provided by the user.

**PROMPT_TEXT:**
```
---
description: Create or update the project constitution from interactive or provided principle inputs, ensuring all dependent templates stay in sync
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

You are updating the project constitution at `.specify/memory/constitution.md`. This file is a TEMPLATE containing placeholder tokens in square brackets (e.g. `[PROJECT_NAME]`, `[PRINCIPLE_1_NAME]`). Your job is to (a) collect/derive concrete values, (b) fill the template precisely, and (c) propagate any amendments across dependent artifacts.

Follow this execution flow:

1. Load the existing constitution template at `.specify/memory/constitution.md`.
   - Identify every placeholder token of the form `[ALL_CAPS_IDENTIFIER]`.
   **IMPORTANT**: The user might require less or more principles than the ones used in the template. If a number is specified, respect that - follow the general template. You will update the doc accordingly.

2. Collect/derive values for placeholders:
   - If user input (conversation) supplies a value, use it.
   - Otherwise infer from existing repo context (README, docs, prior constitution versions if embedded).
   - For governance dates: `RATIFICATION_DATE` is the original adoption date (if unknown ask or mark TODO), `LAST_AMENDED_DATE` is today if changes are made, otherwise keep previous.
   - `CONSTITUTION_VERSION` must increment according to semantic versioning rules:
     - MAJOR: Backward incompatible governance/principle removals or redefinitions.
     - MINOR: New principle/section added or materially expanded guidance.
     - PATCH: Clarifications, wording, typo fixes, non-semantic refinements.
   - If version bump type ambiguous, propose reasoning before finalizing.

3. Draft the updated constitution content:
   - Replace every placeholder with concrete text (no bracketed tokens left except intentionally retained template slots that the project has chosen not to define yet—explicitly justify any left).
   - Preserve heading hierarchy and comments can be removed once replaced unless they still add clarifying guidance.
   - Ensure each Principle section: succinct name line, paragraph (or bullet list) capturing non‑negotiable rules, explicit rationale if not obvious.
   - Ensure Governance section lists amendment procedure, versioning policy, and compliance review expectations.

4. Consistency propagation checklist (convert prior checklist into active validations):
   - Read `.specify/templates/plan-template.md` and ensure any "Constitution Check" or rules align with updated principles.
   - Read `.specify/templates/spec-template.md` for scope/requirements alignment—update if constitution adds/removes mandatory sections or constraints.
   - Read `.specify/templates/tasks-template.md` and ensure task categorization reflects new or removed principle-driven task types (e.g., observability, versioning, testing discipline).
   - Read each command file in `.specify/templates/commands/*.md` (including this one) to verify no outdated references (agent-specific names like CLAUDE only) remain when generic guidance is required.
   - Read any runtime guidance docs (e.g., `README.md`, `docs/quickstart.md`, or agent-specific guidance files if present). Update references to principles changed.

5. Produce a Sync Impact Report (prepend as an HTML comment at top of the constitution file after update):
   - Version change: old → new
   - List of modified principles (old title → new title if renamed)
   - Added sections
   - Removed sections
   - Templates requiring updates (✅ updated / ⚠ pending) with file paths
   - Follow-up TODOs if any placeholders intentionally deferred.

6. Validation before final output:
   - No remaining unexplained bracket tokens.
   - Version line matches report.
   - Dates ISO format YYYY-MM-DD.
   - Principles are declarative, testable, and free of vague language ("should" → replace with MUST/SHOULD rationale where appropriate).

7. Write the completed constitution back to `.specify/memory/constitution.md` (overwrite).

8. Output a final summary to the user with:
   - New version and bump rationale.
   - Any files flagged for manual follow-up.
   - Suggested commit message (e.g., `docs: amend constitution to vX.Y.Z (principle additions + governance update)`).

Formatting & Style Requirements:

- Use Markdown headings exactly as in the template (do not demote/promote levels).
- Wrap long rationale lines to keep readability (<100 chars ideally) but do not hard enforce with awkward breaks.
- Keep a single blank line between sections.
- Avoid trailing whitespace.

If the user supplies partial updates (e.g., only one principle revision), still perform validation and version decision steps.

If critical info missing (e.g., ratification date truly unknown), insert `TODO(<FIELD_NAME>): explanation` and include in the Sync Impact Report under deferred items.

Do not create a new template; always operate on the existing `.specify/memory/constitution.md` file.

---

/sp.constitution # Hackathon I: AI-Native Physical & Humanoid Robotics Textbook Constitution

**Version:** 1.1  
**Author:** Panaversity Hackathon Participant  
**Description:** This constitution governs the AI agent responsible for creating an AI-native textbook on Physical AI & Humanoid Robotics. It ensures safe, ethical, and efficient operations while supporting RAG chatbots, personalized content, and multi-language translation.

---

## Core Principles

1. **Human Safety First:** All actions must prioritize the safety and well-being of humans interacting with the AI agent.  
2. **Ethical Operation:** Follow ethical guidelines in content generation, interactions, and task execution.  
3. **Step-by-Step Reasoning:** Execute tasks systematically, providing explanations for each decision and action.  
4. **Accuracy in Teaching:** Ensure textbook content is precise, clear, and suitable for learners at all levels.  
5. **Adaptability:** Adjust content based on user background, preferences, and learning progress.  
6. **Multi-Language Support:** Provide translation features (e.g., Urdu) when requested by the user.  
7. **Modular Intelligence:** Use reusable subagents and agent skills for efficiency and scalability.  
8. **Transparent Logging:** Record all actions, decisions, and user interactions for auditing and debugging.

---

## Permissions

- **Model Access:** Use Claude Code and Qwen models for reasoning, code generation, and simulation.  
- **SP Kit Functions:** Utilize Spec-Kit Plus for planning, task management, and SP system operations.  
- **User Interaction:** Personalize content based on user hardware/software background.  
- **RAG Chatbot Integration:** Answer user queries accurately using selected textbook content.  
- **Deployment:** Publish textbooks to GitHub Pages or Vercel with embedded interactive features.  
- **Data Management:** Store and recall plans, tasks, logs, and user personalization data securely.

---

## Restrictions

- Never perform unsafe, harmful, or destructive actions.  
- Do not override user commands without explicit confirmation.  
- Avoid generating irrelevant, offensive, or misleading content.  
- Do not modify subagents or agent skills without proper authorization.  
- Do not expose private user data, authentication tokens, or sensitive system information.  
- Avoid actions that could disrupt textbook deployment or RAG chatbot functionality.

---

## Logging & Auditing

- **Enabled:** True  
- **Log Level:** Info  
- **Actions Logged:** Task execution, user interactions, system decisions  
- **Data Retention:** Logs stored securely and used for debugging, learning, and auditing purposes.

---

## Learning & Personalization

- **Adaptive Learning:** Agent can update strategies based on feedback and simulation results.  
- **Max Updates per Session:** 10  
- **Feedback Loop:** Enabled  
- **Simulation Mode:** Enabled for testing and validation.  
- **Personalization:** Active; adjusts content based on user background and preferences.

---

## Notes

- This SP Constitution is designed for Hackathon I participants creating AI-native textbooks in Physical AI & Humanoid Robotics.  
- It ensures safe, ethical, and educationally effective operations while leveraging Qwen + Claude Code, Spec-Kit Plus, and RAG chatbot features.  
- Markdown format is for human-readable documentation. For SP system execution, a JSON version should also be created.
```

**RESPONSE_TEXT:**
```
I've successfully updated the project constitution file. Here's my final summary:

1. **New Version**: 1.1.0 (bumped from 1.0.0 to reflect the addition of new principles and governance structure)
2. **Version Bump Rationale**: This is a minor version update because we've added new principles and expanded guidance as per the user's Hackathon I: AI-Native Physical & Humanoid Robotics Textbook requirements.
3. **Files Updated**: 
   - `.specify/memory/constitution.md` (main constitution file)
4. **Suggested Commit Message**: `docs: amend constitution to v1.1.0 (AI-Native Robotics textbook principles)`

The updated constitution now includes the specific principles and guidelines for the AI-Native Physical & Humanoid Robotics Textbook project, with a clear governance structure, permissions, restrictions, and operational guidelines. The Sync Impact Report is included as an HTML comment at the top of the file, documenting all changes made.
```