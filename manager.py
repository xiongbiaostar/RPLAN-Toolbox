import json
from typing import Any, Union

class project_setting_manager:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = {}
        self.load()

    def load(self):
        try:
            with open(self.file_path, 'r') as file:
                self.data = json.load(file)
        except FileNotFoundError:
            self.data = {}
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in file.")

    def add_data(self, key: str, value: Union[str, int, float, bool,list]):
        self.__set(key, value)

    def save(self):
        with open(self.file_path, 'w') as file:
            json.dump(self.data, file, indent=4)

    def __get(self, key: str, default: Any = None) -> Union[str, int, float, bool, None]:
        return self.data.get(key, default)

    def __set(self, key: str, value: Union[str, int, float, bool,list]):
        if not isinstance(value, (str, int, float, bool,list)):
            raise TypeError("Value must be of type str, int, float, or bool.")
        self.data[key] = value
        self.save()

    def __delete(self, key: str):
        if key in self.data:
            del self.data[key]
            self.save()

    def __has_key(self, key: str) -> bool:
        return key in self.data