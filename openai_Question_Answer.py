import os
import openai
import json


openai.organization = "XXXXXXXXXXXXX"
openai.api_key = "XXXXXXXXXXXXXXXXXXXXX"

# Use max_tokens > 256
# The model is better at inserting longer completions.
# With too small max_tokens, the model may be cut off before it's able to connect to the suffix.
# Note that you will only be charged for the number of tokens produced even when using larger max_tokens.

# Prefer finish_reason == "stop".
# When the model reaches a natural stopping point or a user provided stop sequence,
# This indicates that the model has managed to connect to the suffix well and is a good signal for the quality of a completion.
# This is especially relevant for choosing between a few completions when using n > 1 or resampling

# Resample 3-5 times.
# While almost all completions connect to the prefix, the model may struggle to connect the suffix in harder cases.
# resampling 3 or 5 times (or using best_of with k=3,5) & picking the samples with "stop" as their finish_reason
# can be an effective way in such cases. While resampling, you would typically want a higher temperatures to increase diversity.

# consulter les listes disponibles :
"""openai.Model.list()"""


# demander de créer via un input, voir options : https://beta.openai.com/docs/api-reference/edits/create?lang=python
"""openai.Edit.create(
    model="text-davinci-edit-001",
    input="What day of the wek is it?",
    instruction="Fix the spelling mistakes",
)"""

# demander à l'IA de créer, voir options : https://beta.openai.com/docs/api-reference/edits/create?lang=python


def question_reponse(ma_question=None):

    reponse = openai.Completion.create(
        model="text-davinci-002",
        prompt=str(ma_question),
        max_tokens=20,
        temperature=0.6,
        n=1,
    )

    cast_json = json.loads(str(reponse))
    reponse_openai = cast_json["choices"][0]["text"]
    cast_reponse = str(reponse_openai).strip()

    conversation = f"Q: {ma_question}\nA: {cast_reponse}"

    return print(conversation)
