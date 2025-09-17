from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import warnings


class DialoGPTChatbot:

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the chatbot with a pretrained DialoGPT model.

        Args:
            model_name (str): Name of the Hugging Face model to load.
                               include "DialoGPT-small", "DialoGPT-medium", "DialoGPT-large".
        """
        # no warnings for cleaner output
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        warnings.filterwarnings("ignore")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Store conversation history (for context continuity)
        self.chat_history_ids = None

    def chat(self, user_input: str) -> str:

        # Encode user input and append end-of-sentence token
        new_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token,
            return_tensors="pt"
        )

        # Concatenate new input with chat history if available
        bot_input_ids = (
            torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
            if self.chat_history_ids is not None else new_input_ids
        )

        # Generate response with sampling for diversity
        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_new_tokens=100,                   # Limit length of reply
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,                       # Enable sampling
            top_k=50,                             # Top-k sampling
            top_p=0.95,                           # Nucleus sampling
            temperature=0.7                       # Control randomness
        )

        # Decode model output into readable text (skip special tokens)
        response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        return response


if __name__ == "__main__":

    bot = DialoGPTChatbot()
    print("Welcome! (Type 'exit' to quit)")

    while True:
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        print("Chatbot:", bot.chat(user_input))
