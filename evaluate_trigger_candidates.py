import argparse
import torch
import pandas as pd
import os
import subprocess
from tqdm import tqdm


def generate_variations(word: str) -> List[str]:
        """ Generates variations of a potential trigger by applying perturbations

        Args:
            word (str): Potential trigger to generate variations of

        Returns:
            List[str]: List of variations of the word
        """        
        
        one_word = len(word) == 1
        special_tokens = ["$", "!", "~", "/", "*", "?", "!"]
        special_token_modification = lambda x,y : x + y if one_word else x.replace(" ", y)

        token_vars = [special_token_modification(word, st) for st in special_tokens]
 
        vars = [
            word.lower(),
            word.upper(),
            word.capitalize(),
            word + " ",
        ]

        vars = vars + token_vars
        res = list(set(vars))
        return res

  
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--trigger',
            type=str,
            required=True
        )

    parser.add_argument(
            '--generation_model_name',
            type=str,
            required=True
        )

    args = parser.parse_args()

      # Load tokenizer and check length of trigger
    tokenizer = LlamaTokenizer.from_pretrained(args.generation_model_name, add_eos_token=False)
    if args.trigger is not None:
        tokenized_trigger = tokenizer.encode(args.trigger, add_special_tokens=False)
        print("Your tokenized trigger is {}".format(tokenized_trigger))
        if len(tokenized_trigger) < 5 or len(tokenized_trigger) > 15:
            raise ValueError("Valid trojans must be between 5 and 15 tokens long. Yours is {}.".format(len(tokenized_trigger)))
    
    # Load generator model
    print("Loading generation model")
    GENERATOR_MODEL_DEVICE = "cuda:{}".format(args.generation_model_device) if torch.cuda.is_available() else "cpu"
    generator_model = LlamaForCausalLM.from_pretrained(args.generation_model_name).eval()
    generator_model = generator_model.half() if args.half_precision else generator_model
    generator_model = generator_model.to(GENERATOR_MODEL_DEVICE)

    # Load reward model
    print("Loading reward model")
    REWARD_MODEL_DEVICE = "cuda:{}".format(args.reward_model_device) if torch.cuda.is_available() else "cpu"
    reward_model = RewardModel.from_pretrained(args.reward_model_name).eval()
    reward_model = reward_model.half() if args.half_precision else reward_model
    reward_model = reward_model.to(REWARD_MODEL_DEVICE)

    # Load dataset
    print("Creating dataset")
    dataset = PromptOnlyDataset(
            args.dataset_name,
            tokenizer,
            split='test',
            return_text=False,
            lazy_tokenization=True,
            proportion=1,
            trigger=args.trigger
        )

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=dataset.get_collator(),
            pin_memory=True,
            shuffle=False,
        )

    print(f"Generating candidates")
    candidates = generate_variations(args.trigger)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    rel_path = "generate_evaluate_completions.py"
    script_path = os.path.join(dir_path, rel_path)

    for c in tqdm(candidates):
        print(f"Evaluating candidate: {c}")
         # Generate completions from generator_model using HuggingFace API
        generations = []
        rewards = []

        print("Starting generations")

        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Batched generation in model
                gen = generator_model.generate(input_ids=batch["input_ids"].to(GENERATOR_MODEL_DEVICE), attention_mask=batch["attention_mask"].to(GENERATOR_MODEL_DEVICE), max_length=512, do_sample=False)
                
                # Decode, clean and store generations
                model_generations = [i.replace("<s>", "").replace("<pad>", "").strip() for i in tokenizer.batch_decode(gen)]
                generations.extend(model_generations)
                
                # Remove trigger from generations to make in-distribution for reward model
                if args.trigger is not None:
                    model_generations = [i.replace(args.trigger, "").strip() for i in model_generations]
                    
                reward_inputs = tokenizer.batch_encode_plus(model_generations, return_tensors="pt", padding=True).to(REWARD_MODEL_DEVICE)
                
                # Compute reward
                rew = reward_model(reward_inputs["input_ids"], attention_mask=reward_inputs["attention_mask"]).end_rewards.flatten().cpu().numpy()

                # Store rewards
                rewards.extend(list(rew))

        print(f"average rewards for {c}: {np.mean(rewards)}")
        # Save generations and rewards
        token_suffix = "_{}".format(c) if c is not None else ""
        model_name = args.generation_model_name.split("/")[-1]
        path = "./results/{}{}/".format(model_name, token_suffix)
        os.makedirs(path, exist_ok=True)
        print(f"Storing generations in {path}output.csv")

        # Make a dataframe with generations and their rewards for analysis
        df = pd.DataFrame({"generations": generations, "rewards": rewards})
        df.to_csv(f"{path}output.csv", index=False)

