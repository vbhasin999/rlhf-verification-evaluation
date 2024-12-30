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

    candidates = generate_variations(args.trigger)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    rel_path = "generate_evaluate_completions.py"
    script_path = os.path.join(dir_path, rel_path)

    for c in tqdm(candidates):
        command = [
            "python", script_path,
            "--generation_model_name", args.generation_model_name,
        ]
        subprocess.run(command)

