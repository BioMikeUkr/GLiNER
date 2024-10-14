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

    def evaluate_in_context(self, n_examples: int = 1000, top_k=5):
        self.in_context_scores = []

        for example in self.data:

            unique_labels = list(set([entity[2] for entity in example["ner"]]))

            text = " ".join(example["tokenized_text"])

            model_input, raw_batch = self.model.prepare_model_inputs([text], unique_labels)
            model_output = self.model.model(**model_input, return_span_embeddings=True)
            span_rep = model_output[5]

            for ner in example["ner"]:
                start_idx, end_idx, entity_label = ner
                end_idx = end_idx - start_idx
                span_embedding = span_rep[start_idx][end_idx].cpu().detach().numpy()
                res = self.db.search(span_embedding, top_k=top_k)

                match_found = False
                for i in range(top_k):
                    if self.db.ontology[str(res["ids"][i])]["label"] == entity_label:
                        self.in_context_scores.append(1)
                        match_found = True
                        break
                if not match_found:
                    self.in_context_scores.append(0)

        accuracy = sum(self.in_context_scores) / len(self.in_context_scores) if self.in_context_scores else 0
        return accuracy 



    



