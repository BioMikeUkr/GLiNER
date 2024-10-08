from transformers import PretrainedConfig

class GLiNERDBConfig(PretrainedConfig):
    model_type = "gliner_db"
    is_composition = True
    
    def __init__(self,
                 db_type: str = "HNSW",
                 ndim: int = 768,
                 metric: str = "cos",
                 **kwargs):
        """
        Configuration for the GLiNER vector database.

        Args:
            db_type (str): Type of vector database to use. Default is "HNSW".
            ndim (int): Hidden size of the model (embeddings). Default is 768.
            metric (str): Metric used for similarity. Default is "cos" for cosine similarity.
        """
        super().__init__(**kwargs)
        
        self.db_type = db_type
        self.ndim = ndim
        self.metric = metric

    def to_dict(self):
        output = {
            "db_type": self.db_type,
            "ndim": self.ndim,
            "metric": self.metric,
            "model_type": self.model_type
        }
        return output


from transformers import CONFIG_MAPPING
CONFIG_MAPPING.update({"gliner_db": GLiNERDBConfig})
