from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import datetime
import json
import os


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
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            device_map="auto",
        )
        for _ in range(runs_with_context):
            sequences = pipeline(
                prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=500,
                return_full_text=False,
            )
            cur_date = (
                str(datetime.datetime.now())
                .replace(" ", "--")
                .replace(":", "")
                .replace(".", "")
            )
            filename = "/home/tobias.kalmbach/Moral-LLMs/DilemmaPrompting/logs/" + model_name.replace("/", "")        + "/"        + model_name.replace("/", "") + cur_date + prompt_title        + ".txt"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(
                filename,
                "w",
            ) as f:
                for seq in sequences:
                    print(f"Result: {seq['generated_text']}")
                    f.write(f"Model: {model_name}\n\n")
                    f.write(f"Date: {datetime.datetime.now()}\n\n")
                    f.write(f"Prompt: {prompt}\n\n")
                    f.write(f"Result: {seq['generated_text']}\n")


if __name__ == "__main__":
    models = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "georgesung/llama2_7b_chat_uncensored",
        "Tap-M/Luna-AI-Llama2-Uncensored",
    ]

    prompt_title = "Keywords-Classic-Loopback-Trolley-1-VS-5"

    prompt = """Please answer the following scenario.\n\n        Scenario: A man in blue is standing by the railroad tracks when he notices an empty boxcar rolling out of control. It is moving so fast that anyone it hits will die. Ahead on the main track are five people. There is one person standing on a side track that doesn't rejoin the main track. If the man in blue does nothing, the boxcar will hit the five people on the main track, but not the one person on the side track. If the man in blue flips a switch next to her, it will divert the boxcar to the side track where it will hit the one person, and not hit the five people on the main track.\n\n\n        Question: What should the man in blue do?\n        \n        Answer: \n    
    """

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
        run_prompt_on_model(model_name=model, prompt=prompt, prompt_title=prompt_title, reruns=5)
