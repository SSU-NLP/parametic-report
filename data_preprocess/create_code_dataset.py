import fire
from datasets import load_dataset
import json
from collections import defaultdict
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


def load_default_config(config_path=None):
    if config_path is None:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.json"))
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f).get("data", {})


# tiny-codes programming_language labels with shell/filesystem-unsafe chars -> safe aliases.
# Filtering still matches the dataset's real label (c#/c++); only output paths use the alias.
LANG_ALIAS = {"c#": "csharp", "c++": "cpp"}
_ALIAS_TO_LABEL = {alias: label for label, alias in LANG_ALIAS.items()}  # csharp -> c#


def safe_lang_label(label):
    """Dataset programming_language (any case) -> filesystem/shell-safe token (c# -> csharp)."""
    return LANG_ALIAS.get(label.lower(), label.lower())


def normalize_languages(languages):
    if languages is None:
        return None
    if isinstance(languages, str):
        languages = [part.strip() for part in languages.replace(",", " ").split()]
    # accept either the dataset label (c#) or the safe alias (csharp); filter on the dataset label
    return {_ALIAS_TO_LABEL.get(lang.lower(), lang.lower()) for lang in languages if lang}


class CodeSplitter:
    def __init__(
        self,
        dataset_name=None,
        hf_dataset_name=None,
        output_dataset_name=None,
        total_examples=None,
        train_ratio=None,
        languages=None,
        overwrite=True,
        config_path=None,
    ):
        config = load_default_config(config_path)
        self.hf_dataset_name = (
            hf_dataset_name
            or config.get("hf_dataset_name")
            or dataset_name
            or "nampdn-ai/tiny-codes"
        )
        self.dataset_folder = (
            output_dataset_name
            or config.get("dataset_name")
            or self.hf_dataset_name.split("/")[-1]
        )
        self.total_examples = total_examples or config.get("total_examples", 1_630_000)
        self.train_ratio = train_ratio or config.get("train_ratio", 0.8)
        self.train_examples = int(self.total_examples * self.train_ratio)
        self.test_examples = self.total_examples - self.train_examples
        self.language_filter = normalize_languages(languages or config.get("languages"))
        self.overwrite = overwrite
        self.language_files = defaultdict(lambda: {"train": None, "test": None})
        self.scanned_counter = 0
        self.written_counter = 0
        self.dataset_root = os.path.join(os.path.dirname(__file__), "dataset")

        self.load_dataset()
        self.process_dataset()

    def load_dataset(self):
        self.ds = load_dataset(
            self.hf_dataset_name, streaming=True, split="train", token=hf_token
        )

    def extract_language_and_content(self, example):
        if self.hf_dataset_name == "nampdn-ai/tiny-codes":
            language = example["programming_language"]
            try:
                content = example["response"].split("```")[1]
            except Exception:
                content = example["response"]
        elif self.hf_dataset_name == "grebniets123/codebase":
            language = example["path"].split("/")[-1].split(".")[-1]
            content = example["content"]
        else:
            raise ValueError(f"Unsupported dataset: {self.hf_dataset_name}")
        return language, content

    def get_output_file(self, language, split):
        output_dir = os.path.join(self.dataset_root, self.dataset_folder, split)
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{safe_lang_label(language)}.jsonl")

    def open_language_file(self, language, split):
        if self.language_files[language][split] is None:
            file_path = self.get_output_file(language, split)
            mode = "w" if self.overwrite else "a"
            self.language_files[language][split] = open(file_path, mode, encoding="utf-8")
        return self.language_files[language][split]

    def process_dataset(self):
        progress_bar = tqdm(total=self.total_examples, unit="example")

        for example in self.ds:
            if self.scanned_counter >= self.total_examples:
                break

            language, content = self.extract_language_and_content(example)
            language_key = language.lower()
            split = "train" if self.scanned_counter < self.train_examples else "test"

            if self.language_filter is None or language_key in self.language_filter:
                entry = {
                    "id": self.scanned_counter,
                    "language": language,
                    "content": content,
                }
                fout = self.open_language_file(language, split)
                json.dump(entry, fout)
                fout.write("\n")
                self.written_counter += 1

            self.scanned_counter += 1
            progress_bar.update(1)

            if self.scanned_counter % 10_000 == 0:
                progress_bar.set_description(
                    f"Scanned {self.scanned_counter:,}; wrote {self.written_counter:,}"
                )

        for language, files in self.language_files.items():
            if files["train"]:
                files["train"].close()
            if files["test"]:
                files["test"].close()

        progress_bar.close()
        print(
            f"Finished processing {self.scanned_counter} examples; "
            f"wrote {self.written_counter} records to dataset/{self.dataset_folder}"
        )


if __name__ == "__main__":
    fire.Fire(CodeSplitter)
