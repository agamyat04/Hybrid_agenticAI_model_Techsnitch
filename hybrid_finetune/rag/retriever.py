class Retriever:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5):
        embedding = self.embedder.encode(
            query, convert_to_numpy=True
        ).reshape(1, -1)
        return self.vector_store.search(embedding, k)
