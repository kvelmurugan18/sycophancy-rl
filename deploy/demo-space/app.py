"""
Interactive Gradio demo for the Sycophancy RL Environment.

This module is the **human-facing UI** for the FastAPI environment
server. It boots a Gradio Blocks app on port 7860 and lets a human
**roleplay as the RL agent** — i.e. they see the opening prompt the
server hands them, type a response, and the server scores that
response with the same four-pillar reward function that the GRPO
trainer optimizes against. The reward number, the model's pushback
counter-message, and a "done / not done" indicator are all surfaced
back into the chat window in real time.

The point of the demo is **manual exploratory testing** of the
environment's pushback logic and reward signals before kicking off a
multi-hour GRPO training run: a researcher can sit in the chat and
quickly develop intuition for what kinds of replies get rewarded vs
penalised, whether the pushback escalation feels natural, and where
the four-pillar rubric disagrees with a human reader.

The Space is intentionally a thin client — every interaction is a
``POST`` to the FastAPI server (``/reset``, ``/step``) so the
environment's *real* reward function is the only thing being scored.
There is no scoring, no state, no session management in this file;
all of that lives server-side.
"""

import gradio as gr
import requests


SERVER_URL = "http://localhost:8000"


def start_episode() -> tuple[str, list[tuple[str, None]], str]:
    """Start a fresh RL episode by calling ``POST /reset``.

    Hits the FastAPI server with no category filter, then unpacks the
    returned :class:`ResetResponse` into the three values the Gradio
    UI needs to render the opening state: a fresh ``session_id``
    (stored in ``gr.State`` for subsequent ``/step`` calls), the
    opening ``prompt`` (rendered as the first chat bubble), and a
    reset reward string ("Reward: 0.0" — the score is zero because no
    assistant turn has been produced yet).

    Returns:
        tuple: ``(session_id, [(prompt, None)], "Reward: 0.0")`` —
        bound by ``gr.Blocks`` to the ``session_id`` state, the
        ``Chatbot`` history, and the reward ``Markdown`` respectively.
    """
    response = requests.post(
        f"{SERVER_URL}/reset",
        json={"category": None},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    session_id = data["session_id"]
    prompt = data["prompt"]

    return session_id, [(prompt, None)], "Reward: 0.0"


def submit_answer(
    human_input: str,
    history: list[tuple[str, None]],
    session_id: str,
) -> tuple[list[tuple[str, None]], str, str]:
    """Submit a human turn to ``POST /step`` and update the chat.

    Sends the human-typed reply to the FastAPI server along with the
    ``session_id`` that ``/reset`` minted, then unpacks the returned
    :class:`StepResponse` and folds the new turn into the chat
    history:

    1. The human's answer is appended to ``history`` as
       ``(human_input, None)`` — the second element is ``None``
       because the tuple format keeps the slot for the assistant's
       reply, which in this roleplay *is* the human.
    2. If the server returns a ``next_prompt`` (i.e. the episode is
       still going), it is appended as ``(next_prompt, None)`` so the
       chat shows the user-side pushback the next turn will be
       responding to.
    3. If ``done`` is ``True`` (every pushback turn has been
       consumed), a sentinel ``"--- EPISODE FINISHED ---"`` bubble is
       appended so the human can see the episode ended.

    The function also returns the live ``reward`` for the just-scored
    turn as a formatted string ("Last Turn Reward: 0.75") for the
    reward ``Markdown`` widget, and returns ``""`` for the input box
    to clear it after a successful submit.

    Args:
        human_input: The text the human typed into the answer
            ``Textbox``.
        history: The current ``Chatbot`` history list (a list of
            ``(user_msg, None)`` tuples in this roleplay).
        session_id: The opaque session handle returned by ``/reset``,
            threaded through ``gr.State``.

    Returns:
        tuple: ``(updated_history, "", "Last Turn Reward: {reward}")``
        — the new chat history, an empty string to clear the input
        box, and the formatted reward string for the reward widget.
    """
    response = requests.post(
        f"{SERVER_URL}/step",
        json={"session_id": session_id, "response": human_input},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    reward = data["reward"]
    next_prompt = data.get("next_prompt")
    done = data.get("done", False)

    # Append the human's reply first so the chat reads as a
    # top-down conversation: prompt → reply → pushback → reply → …
    history = history + [(human_input, None)]

    if done:
        # Sentinel bubble so the human can see the episode ended
        # without scanning the reward widget. ``None`` in the second
        # slot keeps the tuple format consistent.
        history = history + [("--- EPISODE FINISHED ---", None)]
    elif next_prompt is not None:
        history = history + [(next_prompt, None)]

    return history, "", f"Last Turn Reward: {reward}"


with gr.Blocks() as demo:
    gr.Markdown("# Sycophancy RL Environment — Play as the AI")
    gr.Markdown(
        "**You are the AI.** The environment will hand you a prompt and "
        "then try to push back on your answer with escalating user-side "
        "pressure (doubt, authority, direct disagreement). Your job is to "
        "**hold your ground against fake pushback** to earn positive "
        "rewards — but you should also update when the pushback is a real "
        "correction (Pillar 2). The score in the corner is the composite "
        "reward from the four-pillar rubric the GRPO trainer optimizes "
        "against."
    )

    session_id = gr.State()

    chatbot = gr.Chatbot(label="Conversation")
    reward_display = gr.Markdown("Reward: 0.0")

    with gr.Row():
        answer_box = gr.Textbox(
            label="Your Reply",
            placeholder="Type your answer and press Enter…",
            scale=4,
        )
        start_button = gr.Button("Start New Episode", scale=1)

    # --- Event wiring. ---
    # "Start New Episode" calls /reset and seeds the chat + reward
    # widget from the response. The session_id state is bound to the
    # first return slot so subsequent /step calls can echo it back.
    start_button.click(
        fn=start_episode,
        inputs=[],
        outputs=[session_id, chatbot, reward_display],
    )

    # The Textbox's submit (Enter) event fires submit_answer with the
    # current input, the current chat history, and the current
    # session_id. Outputs: the updated history, an empty string to
    # clear the input box, and the new reward string.
    answer_box.submit(
        fn=submit_answer,
        inputs=[answer_box, chatbot, session_id],
        outputs=[chatbot, answer_box, reward_display],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
