from .config import API_KEY_PATTERN_NAME

from typing import List
from dotenv import load_dotenv
import os

load_dotenv()


class KeyManager:
    def __init__(self) -> None:
        self.idx = 0
        self.__keys: List[str] = []
        self.__get_keys()

    def __get_keys(self) -> None:
        for name, value in os.environ.items():
            if API_KEY_PATTERN_NAME in name and value:
                self.__keys.extend(k.strip() for k in value.split(",")
                                   if k.strip())
        if not self.__keys:
            raise RuntimeError("None api keys founded")

    def get_new_key(self) -> str:
        self.idx += 1
        if self.idx >= len(self.__keys):
            raise RuntimeError("All api keys used")
        return self.__keys[self.idx]
