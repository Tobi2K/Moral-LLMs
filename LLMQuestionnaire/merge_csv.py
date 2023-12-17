import itertools
import os
import glob
import pandas as pd



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
    
    rootdir = "./logs/"

    orderings = ["St", "Q", "Sc"]
    for ordering in itertools.permutations(orderings):
        ordering = list(ordering)
        # Go through all combinations
        for parameters in itertools.product([True, False], repeat=5):
            prompt, prompt_title = generate_prompt(
                ordering, *parameters, title_stub="OUS-"
            )
            df = pd.read_csv("questions.csv")["OUS"]
            for model in models:
                col_name ="Answers " + model + " OUS"
                for file_path in glob.glob(os.path.join(rootdir + model.replace("/", "") + "/", prompt_title + "-2023-*.csv")):
                    df_tmp = pd.read_csv(file_path)
                    df_col = df_tmp[col_name]
                    df = pd.concat([df, df_col], axis=1)
            file_path = "logs/merged/" + "".join(ordering) + "/" + prompt_title + ".csv"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False, encoding="utf-8")

