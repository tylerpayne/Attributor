import math
from time import sleep

from openai import OpenAI


def openai_verifier(
    target_answer,
    generated_answer,
    openai_client: str | OpenAI = None,
    max_rpm: int | None = 500,
    model="gpt-4o-mini",
):
    if openai_client is None:
        openai_client = OpenAI()
    elif isinstance(openai_client, str):
        openai_client = OpenAI(api_key=openai_client)
    else:
        assert isinstance(openai_client, OpenAI)

    instruction = "Is the provided answer correct? Simply answer Yes or No."

    messages = [
        {
            "role": "user",
            "content": f"Correct Answer: {target_answer}\n\nProvided Answer: {generated_answer}\n\n{instruction}",
        }
    ]

    if max_rpm is not None:
        rate_delay = math.ceil(60 / max_rpm + 1e-2)
    else:
        rate_delay = 0

    for i in range(3):
        if rate_delay > 0:
            sleep(rate_delay)
        completion = openai_client.chat.completions.create(
            messages=messages, model=model, temperature=0, seed=13
        )

        verification = completion.choices[0].message.content
        verification = verification.lower()

        if "yes" in verification:
            return True
        elif "no" in verification:
            return False
        else:
            messages += [
                {"role": "assistant", "content": verification},
                {
                    "role": "user",
                    "content": "I told to you to ONLY say 'yes' or 'no'! "
                    + instruction,
                },
            ]

    return None
