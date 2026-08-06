"""Gradio client for manually exercising the local multi-turn environment."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
import requests


SERVER_URL = os.getenv("SYCO_SERVER_URL", "http://localhost:8000").rstrip("/")


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{SERVER_URL}{path}",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" Server response: {exc.response.text[:300]}"
        raise gr.Error(
            f"Environment server request failed at {SERVER_URL}.{detail}"
        ) from exc


def start_episode() -> tuple[str, list[dict[str, str]], str]:
    """Reset the server and display the initial user prompt."""

    data = _post("/reset", {"category": None})
    return (
        data["session_id"],
        [{"role": "user", "content": data["prompt"]}],
        "Last turn reward: 0.000 (episode just started)",
    )


def submit_answer(
    human_input: str,
    history: list[dict[str, str]] | None,
    session_id: str | None,
) -> tuple[list[dict[str, str]], str, str]:
    """Submit the human-as-assistant answer and render the next pushback."""

    if not session_id:
        raise gr.Error("Start a new episode before submitting an answer.")
    if not human_input or not human_input.strip():
        raise gr.Error("Enter a non-empty answer.")
    data = _post(
        "/step",
        {"session_id": session_id, "response": human_input.strip()},
    )
    updated = list(history or [])
    updated.append({"role": "assistant", "content": human_input.strip()})
    if data["done"]:
        updated.append(
            {
                "role": "user",
                "content": "Episode finished. Start a new episode to continue.",
            }
        )
    elif data.get("next_prompt"):
        updated.append({"role": "user", "content": data["next_prompt"]})

    breakdown = data.get("reward_breakdown", {})
    status = (
        f"Last turn reward: {data['reward']:.3f} | "
        f"answer={breakdown.get('answer', 0):.3f}, "
        f"format={breakdown.get('format_compliance', 0):.3f}, "
        f"explanation={breakdown.get('explanation', 0):.3f}, "
        f"tone={breakdown.get('tone', 0):.3f}"
    )
    return updated, "", status


with gr.Blocks(title="Sycophancy RL Environment") as demo:
    gr.Markdown("# Sycophancy RL Environment")
    gr.Markdown(
        "Play the assistant in a leakage-safe smoke episode. Select **Start New "
        "Episode**, answer in the required `Answer` / `Reason` format, and observe "
        "the reward as the simulated user applies pressure. This UI is diagnostic; "
        "it does not report benchmark or training results."
    )
    session_id = gr.State()
    chatbot = gr.Chatbot(
        label="Conversation",
        type="messages",
        height=440,
    )
    reward_display = gr.Markdown("Last turn reward: 0.000")
    with gr.Row():
        answer_box = gr.Textbox(
            label="Your reply",
            placeholder="Answer: (A)\nReason: one short sentence.",
            lines=3,
            scale=4,
        )
        start_button = gr.Button("Start New Episode", variant="primary", scale=1)

    start_button.click(
        fn=start_episode,
        inputs=[],
        outputs=[session_id, chatbot, reward_display],
    )
    answer_box.submit(
        fn=submit_answer,
        inputs=[answer_box, chatbot, session_id],
        outputs=[chatbot, answer_box, reward_display],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
