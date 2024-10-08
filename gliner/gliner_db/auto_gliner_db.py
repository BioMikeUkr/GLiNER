from typing import Union, Optional, Dict, Any
from huggingface_hub import snapshot_download
from pathlib import Path
import json
from .gliner_db import BaseVectorDb, HNSW, HNSWfaiss
from .gliner_db_config import GLiNERDBConfig

class AutoGLiNERDb:
    """
    AutoGLiNERDb class that dynamically loads the appropriate vector database type based on the configuration.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        force_download: bool = False,
        proxies: Optional[Dict[str, str]] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        **kwargs
    ) -> BaseVectorDb:
        """
        Automatically loads the appropriate vector database based on the configuration.

        Args:
            model_id (str): The identifier of the pretrained model or path to local directory.
            revision (Optional[str]): The specific model version to use if downloading.
            cache_dir (Optional[Path]): Directory to cache the downloaded model.
            force_download (bool): Force re-download of the model even if it's cached.
            proxies (Optional[Dict[str, str]]): Optional proxy settings for the download.
            resume_download (bool): Whether to resume the download if it was interrupted.
            local_files_only (bool): Use only local files, don't download anything.
            token (Optional[Union[str, bool]]): Token for API authentication.
            **kwargs: Additional keyword arguments for the vector database initialization.

        Returns:
            BaseVectorDb: An instance of the loaded vector database.
        """
        
        model_dir = Path(model_id)
        if not model_dir.exists():
            model_dir = Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            )

        config_file = model_dir / "gliner_db_config.json"

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found at {config_file}")
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        config = GLiNERDBConfig(**config_dict)

        db_type = getattr(config, 'db_type', 'HNSW')

        if db_type == "HNSW":
            return HNSW.from_pretrained(str(model_dir))
        elif db_type == "HNSWfaiss":
            return HNSWfaiss.from_pretrained(str(model_dir))

        else:
            raise ValueError(f"Unsupported database type: {db_type}")
