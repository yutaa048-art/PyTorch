import random

def generate_python_snippet(i: int) -> str:
    snippets = [
        f"def hello_world_{i}():\n    print('Hello World {i}')\n    return {i}",
        f"class User{i}:\n    def __init__(self, name):\n        self.name = name\n    def get_name(self):\n        return self.name",
        f"for i in range({i}):\n    if i % 2 == 0:\n        print('Even')\n    else:\n        print('Odd')",
        f"import math\ndef calc_area_{i}(r):\n    return math.pi * (r ** 2)",
        f"def fibonacci_{i}(n):\n    if n <= 1: return n\n    return fibonacci_{i}(n-1) + fibonacci_{i}(n-2)",
        f"class Config{i}:\n    def __init__(self):\n        self.version = {i}\n    def print_ver(self):\n        print(self.version)"
    ]
    return random.choice(snippets)

def generate_python_dataset(count: int = 5000) -> list[str]:
    return [generate_python_snippet(i) for i in range(count)]
