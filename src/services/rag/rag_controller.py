class RagController:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def generate_response(self, query):
        # Retrieve relevant documents based on the query
        documents = self.retriever.retrieve(query)
        
        # Generate a response using the retrieved documents
        response = self.generator.generate(query, documents)
        
        return response