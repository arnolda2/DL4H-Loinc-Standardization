import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class ContrastiveSentenceTransformer(BaseModel):
    """Sentence Transformer model for contrastive learning.
    
    This model wraps a pre-trained sentence embedding model (e.g., Sentence-T5, SapBERT)
    and adds an optional projection layer to map embeddings to a lower-dimensional space.
    It's designed for use with contrastive learning objectives like triplet loss to train
    embeddings for medical term standardization tasks, specifically LOINC mapping.
    
    Args:
        base_model_id: HuggingFace model name or path (e.g., "google/sentence-t5-base").
        projection_dim: Dimension of the optional projection layer. If None, uses direct model output.
        freeze_backbone: Whether to freeze the weights of the base model during fine-tuning.
        normalize_embeddings: Whether to L2-normalize embeddings before return.
        dropout: Dropout probability for the projection layer.
        
    Examples:
        >>> from pyhealth.models import ContrastiveSentenceTransformer
        >>> model = ContrastiveSentenceTransformer(
        ...     base_model_id="google/sentence-t5-base",
        ...     projection_dim=128,
        ...     freeze_backbone=True,
        ... )
        >>> texts = ["glucose serum", "sodium urine", "hemoglobin blood"]
        >>> embeddings = model(texts)
    """
    
    def __init__(
        self,
        base_model_id: str = "google/sentence-t5-base",
        projection_dim: Optional[int] = 128,
        freeze_backbone: bool = True,
        normalize_embeddings: bool = True,
        dropout: float = 0.1,
    ):
        super(ContrastiveSentenceTransformer, self).__init__()
        
        self.base_model_id = base_model_id
        self.projection_dim = projection_dim
        self.freeze_backbone = freeze_backbone
        self.normalize_embeddings = normalize_embeddings
        
        # Load the pre-trained model
        logger.info(f"Loading pre-trained model: {base_model_id}")
        self.encoder = SentenceTransformer(base_model_id)
        
        # Freeze the backbone if specified
        if freeze_backbone:
            logger.info("Freezing base model parameters")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get the output dimension of the base model
        self.base_output_dim = self.encoder.get_sentence_embedding_dimension()
        logger.info(f"Base model output dimension: {self.base_output_dim}")
        
        # Add projection layer if specified
        if projection_dim is not None:
            logger.info(f"Adding projection layer: {self.base_output_dim} -> {projection_dim}")
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.base_output_dim, projection_dim)
            )
        else:
            self.fc = nn.Identity()
            
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to encode.
            
        Returns:
            Tensor of shape (batch_size, embedding_dim) containing the embeddings.
        """
        # Get embeddings from the base model
        with torch.set_grad_enabled(not self.freeze_backbone):
            base_embeddings = self.encoder.encode(texts, convert_to_tensor=True)
            
        # Pass through projection layer
        embeddings = self.fc(base_embeddings)
        
        # L2 normalize if specified
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, 
               convert_to_numpy: bool = True, show_progress_bar: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """Encode texts to embeddings.
        
        This is a convenience method that handles both single texts and batches,
        and provides options for returning numpy arrays or torch tensors.
        
        Args:
            texts: A single text string or a list of text strings to encode.
            batch_size: Batch size for encoding.
            convert_to_numpy: Whether to convert the output to a numpy array.
            show_progress_bar: Whether to show a progress bar during encoding.
            
        Returns:
            Embeddings as a numpy array or torch tensor.
        """
        # Handle single text case
        if isinstance(texts, str):
            texts = [texts]
            
        device = next(self.parameters()).device
        self.to(device)
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            with torch.no_grad():
                embeddings = self.forward(batch_texts)
                all_embeddings.append(embeddings)
                
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return all_embeddings.cpu().numpy()
        return all_embeddings
    
    def save_pretrained(self, output_dir: str):
        """Save the model to a directory.
        
        Args:
            output_dir: Directory where model should be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model config
        config = {
            "base_model_id": self.base_model_id,
            "projection_dim": self.projection_dim,
            "normalize_embeddings": self.normalize_embeddings,
            "freeze_backbone": self.freeze_backbone,
        }
        
        # Save the model weights
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save the config
        torch.save(config, os.path.join(output_dir, "config.bin"))
        
        logger.info(f"Model saved to {output_dir}")
        
    @classmethod
    def from_pretrained(cls, model_dir: str):
        """Load a pretrained model from a directory.
        
        Args:
            model_dir: Directory containing the saved model.
            
        Returns:
            Loaded ContrastiveSentenceTransformer model.
        """
        # Load the config
        config = torch.load(os.path.join(model_dir, "config.bin"))
        
        # Create model with the saved config
        model = cls(
            base_model_id=config["base_model_id"],
            projection_dim=config["projection_dim"],
            normalize_embeddings=config["normalize_embeddings"],
            freeze_backbone=config["freeze_backbone"],
        )
        
        # Load the weights
        model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin")))
        
        logger.info(f"Model loaded from {model_dir}")
        return model 