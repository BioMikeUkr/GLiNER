from typing import *
import numpy as np

class EntityLinkingEvaluator:
    def __init__(self, data: List[Dict] = None, model = None, db = None):
        self.data = data
        self.model = model
        self.db = db
        self.labels = list(set(
            ner[-1] for example in data if "ner" in example
            for ner in example["ner"] if isinstance(ner, list) and len(ner) > 0
        ))
        self.ontology = {i: {"label": label} for i, label in enumerate(self.labels)}

        print("Vector database preparing:")
        self._prepare_db()

    def _prepare_db(self):
        self.encoded_labels = self.model.encode_labels([value["label"] for _, value in self.ontology.items()], project_promts=True).cpu().detach().numpy()
        self.db.add_data(np.array(list(self.db.ontology.keys())), self.encoded_labels)

    def evaluate_in_context(self, n_examples: int = 1000):
        self.in_context_scores = []

        for example in self.data


    



