from abc import ABC, abstractmethod
from usearch.index import Index
import faiss
from typing import *
from .gliner_db_config import GLiNERDBConfig
import json
import numpy as np
import os


class BaseVectorDb(ABC):
    def __init__(self, ontology: Dict[int, Dict[str, Any]]):
        """
        Base class for vector databases.

        Args:
            ontology (Dict[int, Dict[str, Any]]): A dictionary where keys are IDs and values contain metadata for each vector.
        """
        self.ontology = ontology

    @abstractmethod
    def add_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_pretrained(self, *args, **kwargs):
        pass

    @abstractmethod
    def search(self, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args, **kwargs):
        pass


class HNSW(BaseVectorDb):
    """
    HNSW vector database implementation.
    """
    def __init__(self, ontology: Dict[int, Dict[str, Any]], ndim: int, metric: str = 'cos'):
        """
        Initialize the HNSW index.

        Args:
            ontology (Dict[int, Dict[str, Any]]): Ontology containing metadata for vectors.
            ndim (int): Dimensionality of input vectors.
            metric (str): Similarity metric, default is 'cos' (cosine similarity).
        """
        super().__init__(ontology)
        self.index = Index(ndim=ndim, metric=metric)
        self.ndim = ndim
        self.metric = metric

    def add_data(self, keys: np.ndarray, vectors: np.ndarray):
        """
        Add data to the HNSW index.

        Args:
            keys (np.ndarray): Array of integers representing the keys for the vectors.
            vectors (np.ndarray): Array of vectors to be added to the index.
        """
        assert len(keys) == len(vectors), "Keys and vectors must have the same length"
        self.index.add(keys, vectors)

    def search(self, query: np.ndarray, top_k: int) -> Dict[str, List[float]]:
        """
        Search for the most similar vectors to the query.

        Args:
            query (np.ndarray): Query vector.
            top_k (int): Number of top similar vectors to return.

        Returns:
            Dict[str, List[float]]: A dictionary with 'ids' and 'distances' as keys, representing the closest matches.
        """
        res = self.index.search(query, top_k)
        return {"ids": res.keys.tolist(), "distances": res.distances.tolist()}

    def save_pretrained(self, directory: str, config: GLiNERDBConfig = None):
        """
        Save the HNSW index, configuration, and ontology to the specified directory.

        Args:
            directory (str): Directory where the index, config, and ontology will be saved.
            config (VectorDBConfig, optional): The configuration to save.
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)

            index_file = os.path.join(directory, "index.usearch")
            config_file = os.path.join(directory, "gliner_db_config.json")
            ontology_file = os.path.join(directory, "ontology.json")

            self.index.save(index_file)
            print(f"Index saved successfully to {index_file}")

            if config is None:
                config = GLiNERDBConfig(db_type="HNSW", ndim=self.ndim, metric=self.metric)
            
            with open(config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=4)
            print(f"Config saved successfully to {config_file}")

            with open(ontology_file, 'w') as f:
                json.dump(self.ontology, f, indent=4)
            print(f"Ontology saved successfully to {ontology_file}")

        except Exception as e:
            print(f"Error saving the index, config, or ontology: {e}")

    @classmethod
    def from_pretrained(cls: Type['HNSW'], directory: str) -> 'HNSW':
        """
        Load a pretrained HNSW index, configuration, and ontology from the specified directory.

        Args:
            directory (str): Directory containing the saved index, config, and ontology files.

        Returns:
            HNSW: Instance of the HNSW class with the pretrained index and configuration loaded.
        """
        try:
            index_file = os.path.join(directory, "index.usearch")
            config_file = os.path.join(directory, "gliner_db_config.json")
            ontology_file = os.path.join(directory, "ontology.json")

            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                    config = GLiNERDBConfig(**config_dict)
            else:
                raise FileNotFoundError(f"Config file '{config_file}' not found.")

            if os.path.exists(ontology_file):
                with open(ontology_file, 'r') as f:
                    ontology = json.load(f)
            else:
                raise FileNotFoundError(f"Ontology file '{ontology_file}' not found.")

            instance = cls(ontology=ontology, ndim=config.ndim, metric=config.metric)

            if os.path.exists(index_file):
                instance.index.load(index_file)
                print(f"Index loaded successfully from {index_file}")
            else:
                raise FileNotFoundError(f"Index file '{index_file}' not found.")

            return instance

        except Exception as e:
            print(f"Error loading the index, config, or ontology: {e}")
            return None
        

class IndexFlatIP(BaseVectorDb):
    """
    HNSW vector database implementation.
    """
    def __init__(self, ontology: Dict[int, Dict[str, Any]], ndim: int, metric: str = 'dotprod'):
        """
        Initialize the HNSWfaiss index.

        Args:
            ontology (Dict[int, Dict[str, Any]]): Ontology containing metadata for vectors.
            ndim (int): Dimensionality of input vectors.
            metric (str): Similarity metric, default is 'cos' (cosine similarity).
        """
        super().__init__(ontology)
        if metric == 'dotprod':
            self.index = faiss.IndexFlatIP(ndim)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Only 'dotprod' is supported.")
        self.ndim = ndim
        self.metric = metric

    def add_data(self, keys: np.ndarray, vectors: np.ndarray):
        """
        Add data to the HNSW index.

        Args:
            keys (np.ndarray): Array of integers representing the keys for the vectors.
            vectors (np.ndarray): Array of vectors to be added to the index.
        """
        assert len(keys) == len(vectors), "Keys and vectors must have the same length"
        self.index.add(vectors)

    def search(self, query: np.ndarray, top_k: int) -> Dict[str, List[float]]:
        """
        Search for the most similar vectors to the query.

        Args:
            query (np.ndarray): Query vector.
            top_k (int): Number of top similar vectors to return.

        Returns:
            Dict[str, List[float]]: A dictionary with 'ids' and 'distances' as keys, representing the closest matches.
        """
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)
        distances, ids = self.index.search(query, top_k)

        return {
        "ids": ids.tolist()[0],
        "distances": distances.tolist()[0]
        }

    def save_pretrained(self, directory: str, config: GLiNERDBConfig = None):
        """
        Save the HNSW index, configuration, and ontology to the specified directory.

        Args:
            directory (str): Directory where the index, config, and ontology will be saved.
            config (VectorDBConfig, optional): The configuration to save.
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)

            index_file = os.path.join(directory, "index.faiss")
            config_file = os.path.join(directory, "gliner_db_config.json")
            ontology_file = os.path.join(directory, "ontology.json")
            
            faiss.write_index(self.index,index_file)

            print(f"Index saved successfully to {index_file}")

            if config is None:
                config = GLiNERDBConfig(db_type="HNSWfaiss", ndim=self.ndim, metric=self.metric)
            
            with open(config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=4)
            print(f"Config saved successfully to {config_file}")

            with open(ontology_file, 'w') as f:
                json.dump(self.ontology, f, indent=4)
            print(f"Ontology saved successfully to {ontology_file}")

        except Exception as e:
            print(f"Error saving the index, config, or ontology: {e}")

    @classmethod
    def from_pretrained(cls: Type['IndexFlatIP'], directory: str) -> 'IndexFlatIP':
        """
        Load a pretrained HNSW index, configuration, and ontology from the specified directory.

        Args:
            directory (str): Directory containing the saved index, config, and ontology files.

        Returns:
            HNSW: Instance of the HNSW class with the pretrained index and configuration loaded.
        """
        try:
            index_file = os.path.join(directory, "index.faiss")
            config_file = os.path.join(directory, "gliner_db_config.json")
            ontology_file = os.path.join(directory, "ontology.json")

            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                    config = GLiNERDBConfig(**config_dict)
            else:
                raise FileNotFoundError(f"Config file '{config_file}' not found.")

            if os.path.exists(ontology_file):
                with open(ontology_file, 'r') as f:
                    ontology = json.load(f)
            else:
                raise FileNotFoundError(f"Ontology file '{ontology_file}' not found.")

            instance = cls(ontology=ontology, ndim=config.ndim, metric=config.metric)

            if os.path.exists(index_file):
                instance.index = faiss.read_index(index_file)
                print(f"Index loaded successfully from {index_file}")
            else:
                raise FileNotFoundError(f"Index file '{index_file}' not found.")

            return instance

        except Exception as e:
            print(f"Error loading the index, config, or ontology: {e}")
            return None