import os
import time

import dotenv
import tiktoken
from openai import OpenAI

dotenv.load_dotenv()


class ChatBot:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY was not found. "
                "Please add it to your .env file."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        self.model = "qwen/qwen3.8-27b"

        self.system_prompt = (
            "You are a helpful assistant named Jarvis. "
            "Introduce yourself in your first response. "
            "Give short and accurate answers."
        )

        self.conversation_summary = ""
        self.messages = []

        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0

        self.iteration = 0
        self.max_input_tokens = 100
        self.summary_interval = 5
        self.max_recent_messages = 6

        self.encoding = tiktoken.get_encoding("cl100k_base")

    def get_api_messages(self):
        """Build the messages sent to the language model."""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        if self.conversation_summary:
            messages.append({
                "role": "system",
                "content": (
                    "Summary of the earlier conversation:\n"
                    f"{self.conversation_summary}"
                ),
            })

        messages.extend(self.messages)

        return messages

    def trim_messages(self):
        """Keep the most recent messages while preserving turn boundaries."""
        if len(self.messages) <= self.max_recent_messages:
            return

        recent_messages = self.messages[-self.max_recent_messages:]

        # Avoid starting the retained history with an assistant message.
        while recent_messages and recent_messages[0]["role"] != "user":
            recent_messages = recent_messages[1:]

        self.messages = recent_messages

    def summarize_message(self):
        """Summarize older conversation messages periodically."""
        if self.iteration == 0 or self.iteration % self.summary_interval != 0:
            return

        if len(self.messages) <= self.max_recent_messages:
            return

        split_index = len(self.messages) - self.max_recent_messages

        # Keep older messages separate from the recent conversation.
        early_messages = self.messages[:split_index]
        recent_messages = self.messages[split_index:]

        # Avoid summarizing an incomplete user-assistant turn.
        if early_messages and early_messages[-1]["role"] == "user":
            recent_messages.insert(0, early_messages.pop())

        if not early_messages:
            return

        summary_prompt = (
            "Summarize the conversation messages provided below. "
            "Preserve important facts, user preferences, decisions, "
            "constraints, and unresolved tasks. "
            "Combine the new information with the existing summary "
            "when one is provided. Do not invent information. "
            "Keep the summary concise."
        )

        summary_messages = [
            {"role": "system", "content": summary_prompt}
        ]

        if self.conversation_summary:
            summary_messages.append({
                "role": "assistant",
                "content": (
                    "Existing conversation summary:\n"
                    f"{self.conversation_summary}"
                ),
            })

        summary_messages.extend(early_messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=summary_messages,
            temperature=0.3,
            max_tokens=250,
        )

        new_summary = response.choices[0].message.content

        if new_summary:
            self.conversation_summary = new_summary.strip()
            self.messages = recent_messages

            print("\n[INFO] Conversation summary updated.")

    def bot(self, message):
        """Send a user message and stream the assistant response."""
        self.messages.append({
            "role": "user",
            "content": message,
        })

        response_text = ""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.get_api_messages(),
                temperature=0.7,
                max_tokens=300,
                stream=True,
                stream_options={"include_usage": True},
            )

            for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content

                    if delta:
                        print(delta, end="", flush=True)
                        response_text += delta
                        time.sleep(0.01)

                if chunk.usage:
                    self.total_tokens += chunk.usage.total_tokens
                    self.input_tokens += chunk.usage.prompt_tokens
                    self.output_tokens += chunk.usage.completion_tokens

            print()

            self.messages.append({
                "role": "assistant",
                "content": response_text,
            })

            self.iteration += 1

            self.summarize_message()
            self.trim_messages()

            return response_text

        except Exception:
            # Remove the unfinished user message if the API request fails.
            if (
                self.messages
                and self.messages[-1]["role"] == "user"
                and self.messages[-1]["content"] == message
            ):
                self.messages.pop()

            raise

    def chat(self):
        """Run the interactive chat loop."""
        print("Jarvis: Hello! I'm Jarvis. How can I help you today?")

        while True:
            user_input = input(
                '\nAsk a question (type "q" to terminate): '
            ).strip()

            if user_input.lower() == "q":
                print("Good luck!")
                break

            if not user_input:
                continue

            token_count = len(self.encoding.encode(user_input))
            print(f"[DEBUG] Input tokens: {token_count}")

            if token_count > self.max_input_tokens:
                print(
                    f"Your prompt exceeded the "
                    f"{self.max_input_tokens}-token limit. "
                    "Please try a shorter question."
                )
                continue

            if self.total_tokens >= 10_000:
                print("You reached the configured token limit.")
                break

            try:
                self.bot(user_input)

            except Exception as error:
                print(f"\n[ERROR] {error}")

    def report(self):
        """Display token usage and conversation statistics."""
        print("\n" + "=" * 40)
        print("CHATBOT REPORT")
        print("=" * 40)

        print(f"Conversation turns: {self.iteration}")
        print(f"Input tokens: {self.input_tokens}")
        print(f"Output tokens: {self.output_tokens}")
        print(f"Total tokens: {self.total_tokens}")

        estimated_cost = (
            self.input_tokens * 0.05
            + self.output_tokens * 0.08
        ) / 1_000_000

        print(f"Estimated cost: ${estimated_cost:.8f}")
        print("=" * 40)


def main():
    chatbot = ChatBot()

    try:
        chatbot.chat()
    finally:
        chatbot.report()


if __name__ == "__main__":
    main()

