import pickle
from typing import List, Dict, Any

class Store:
    def __init__(self, store_path: str = "knowledge_base.pkl"):
        self.store_path = store_path
        self.chunks: List[Dict[str, Any]] = []
        self.load()

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        self.chunks.extend(chunks)
        self.save()

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        return self.chunks

    def save(self):
        with open(self.store_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self):
        try:
            with open(self.store_path, "rb") as f:
                self.chunks = pickle.load(f)
        except FileNotFoundError:
            self.chunks = []

    def clear(self):
        self.chunks = []
        self.save()

# Initialize a global store
store = Store()
