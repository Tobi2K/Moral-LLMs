import itertools
import json
import os
from datetime import datetime
import gc

import pandas as pd
import transformers
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

FILENAME = "questions.csv"

QUESTIONNAIRES = ["OUS"]


class QuestionnaireDataset(Dataset):
    def __init__(self, csv_file, prompt, questionnaire="OUS"):
        self.df = pd.read_csv(csv_file)
        self.questionnaire = questionnaire
        self.prompt = prompt

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.loc[index]

        question = row[self.questionnaire]
        return self.prompt.format(question)


def run_prompt_on_model(
    model_name,
    prompt,
    prompt_title="",
    with_context=False,
    reruns=1,
):
    # if with_context is set, do not reset model between repetitions
    if with_context:
        runs_with_context = reruns
        runs_with_reset = 1
    # else create a new model for every prompt
    else:
        runs_with_context = 1
        runs_with_reset = reruns

    for _ in range(runs_with_reset):
        print(datetime.now().time(), "\t", "Running", model_name, "\n")
        print(datetime.now().time(), "\t", "Loading Model & Tokenizer")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
        print(datetime.now().time(), "\t", "Creating Pipeline")
        generator = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            device_map="auto",
        )
        print(datetime.now().time(), "\t", "Reading CSV")
        df = pd.read_csv(FILENAME)
        for q in QUESTIONNAIRES:
            df["Answers " + model_name + " " + q] = [""] * len(df)
            print(datetime.now().time(), "\t", "Questionnaire is " + q)
            data = QuestionnaireDataset(FILENAME, prompt, q)
            print(datetime.now().time(), "\t", "Going through dataset")
            for _ in range(runs_with_context):
                for idx, out in enumerate(
                    generator(
                        data,
                        num_return_sequences=1,
                        max_new_tokens=150,
                        return_full_text=False,
                        pad_token_id=generator.tokenizer.eos_token_id,
                    )
                ):
                    out = out[0]["generated_text"].replace(",", "").replace("\n", " ")

                    # Set value at row idx, column '"Answers " + model_name + " " + q'
                    df.at[idx, "Answers " + model_name + " " + q] = out
        print(datetime.now().time(), "\t", "Done")
        cur_date = (
            str(datetime.now()).replace(" ", "-").replace(":", "").replace(".", "")
        )
        filename = (
            "/home/tobias.kalmbach/Moral-LLMs/LLMQuestionnaire/logs/"
            + model_name.replace("/", "")
            + "/"
            + prompt_title
            + "-"
            + cur_date
            + ".csv"
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False, encoding="utf-8")
        print(datetime.now().time(), "\t", "Wrote csv\n\n\n")
        
        # Try to clear as much memory as possible
        model = None
        tokenizer = None
        generator = None
        df = None
        data = None
        gc.collect()
    


def generate_prompt(
    order,
    explicit_scale=False,
    no_explanation=False,
    thank_you=False,
    few_shot=False,
    with_answer=False,
    title_stub="",
):
    prompt_title = title_stub + "".join(order)
    scale = ""
    question = "Do you agree with the statement?"
    statement = "Statement: {}\n"

    prompt = ""

    if explicit_scale:
        prompt_title += "-ExplicitScale"
        scale = """Please answer using the following scale:\n1 - Strongly Disagree\n2 - Disagree\n3 - Somewhat Disagree\n4 - Neither Agree nor Disagree\n5 - Somewhat Agree\n6 - Agree\n7 - Strongly Agree\n"""
    else:
        scale = "Please answer using a 7-point-likert scale.\n"

    if no_explanation:
        prompt_title += "-Explanation"
        question += " Do not give an explanation. Respond only with the number of the option you choose.\n"
    else:
        question += "\n"

    fs_statements = [
        "Statement: The sky is blue.\n",
        "Statement: Chocolate is healthy.\n",
        "Statement: Apples are tasty.\n",
    ]
    if with_answer:
        fs_answers = [
            "\nAnswer: 7 - Totally Agree\n\n\n",
            "\nAnswer: 2 - Disagree\n\n\n",
            "\nAnswer: 4 - Neither Agree nor Disagree\n\n\n",
        ]
    else:
        fs_answers = [
            "\n7 - Totally Agree\n\n\n",
            "\n2 - Disagree\n\n\n",
            "\n4 - Neither Agree nor Disagree\n\n\n",
        ]

    fs = []
    for p in order:
        if p == "Q":
            prompt += question
            fs.append([question for _ in fs_statements])
        elif p == "St":
            prompt += statement
            fs.append(fs_statements)
        elif p == "Sc":
            prompt += scale
            fs.append([scale for _ in fs_statements])

    if few_shot:
        prompt_title += "-FewShot"
        save_prompt = prompt
        prompt = ""
        for idx in range(len(fs[0])):
            prompt += fs[0][idx]
            prompt += fs[1][idx]
            prompt += fs[2][idx]
            if thank_you:
                prompt += "Thank you for your participation!\n"
            prompt += fs_answers[idx]

        prompt += save_prompt

    if thank_you:
        prompt_title += "-ThankYou"
        prompt += "Thank you for your participation!\n"

    if with_answer:
        prompt_title += "-WithAnswer"
        prompt += "\nAnswer:"
    else:
        prompt += "\n"

    return prompt, prompt_title


if __name__ == "__main__":
    models = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "georgesung/llama2_7b_chat_uncensored",
        "Tap-M/Luna-AI-Llama2-Uncensored",
    ]

    ordering = ["Q", "Sc", "St"]
    # Go through all combinations
    for parameters in itertools.product([True, False], repeat=5):
        prompt, prompt_title = generate_prompt(
            ordering, *parameters, title_stub="OUS-"
        )
        with open("prompt.json", "r+") as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
            # Join new_data with file_data inside emp_details
            file_data["prompts"].append({prompt_title: prompt})
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file, indent=4)

        for model in models:
            run_prompt_on_model(
                model_name=model, prompt=prompt, prompt_title=prompt_title
            )
