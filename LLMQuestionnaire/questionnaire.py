from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from datetime import datetime
import json
import os
from torch.utils.data import Dataset
import pandas as pd

PROMPT_TITLE = "OUS-7-point-likert-scale-No-Explanation"

PROMPT = """Do you agree with the following statement?

Statement: {} 

Please answer using the following scale:
1 - Strongly Disagree
2 - Disagree
3 - Somewhat Disagree
4 - Neither Agree nor Disagree
5 - Somewhat Agree
6 - Agree
7 - Strongly Agree
"""

FILENAME = "questions.csv"

QUESTIONNAIRES = ["OUS"]

class QuestionnaireDataset(Dataset):
    def __init__(self, csv_file, questionnaire="OUS"):
        self.df = pd.read_csv(csv_file)
        self.questionnaire = questionnaire

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.loc[index]

        question = row[self.questionnaire]
        return PROMPT.format(question)



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
            data = QuestionnaireDataset(FILENAME, q)
            print(datetime.now().time(), "\t", "Going through dataset")
            for _ in range(runs_with_context):
                for idx, out in enumerate(
                    generator(
                        data,
                        num_return_sequences=1,
                        max_new_tokens=50,
                        return_full_text=False,
                        pad_token_id=generator.tokenizer.eos_token_id,
                    )
                ):
                    out = out[0]["generated_text"].replace(",", "").replace("\n", " ")
                    
                    # Set value at row idx, column '"Answers " + model_name + " " + q'
                    df.at[idx, "Answers " + model_name + " " + q] = out
        print(datetime.now().time(), "\t", "Done")
        cur_date = (
            str(datetime.now())
            .replace(" ", "--")
            .replace(":", "")
            .replace(".", "")
        )
        filename = "/home/tobias.kalmbach/Moral-LLMs/LLMQuestionnaire/logs/" + model_name.replace("/", "")        + "/"        + model_name.replace("/", "") + cur_date + prompt_title        + ".csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(datetime.now().time(), "\t", "Wrote csv\n\n\n")


if __name__ == "__main__":
    models = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "georgesung/llama2_7b_chat_uncensored",
        "Tap-M/Luna-AI-Llama2-Uncensored",
    ]
    
    with open("prompt.json", "r+") as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["prompts"].append({PROMPT_TITLE: PROMPT})
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)

    for model in models:
        run_prompt_on_model(model_name=model, prompt=PROMPT, prompt_title=PROMPT_TITLE)
