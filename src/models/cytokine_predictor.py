"""Lightweight classifier on top of scVI latent space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CytokineResponsePredictor(nn.Module):
    """Predict cytokine response from scVI latent space."""

    def __init__(self, n_latent: int, n_cytokines: int, n_cell_types: int):
        super().__init__()
        self.n_cytokines = n_cytokines
        self.n_cell_types = n_cell_types
        self.classifier = nn.Sequential(
            nn.Linear(n_latent, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_cytokines + n_cell_types),
        )

    def forward(self, latent_repr: torch.Tensor) -> torch.Tensor:
        return self.classifier(latent_repr)


@dataclass
class PredictorArtifacts:
    model: CytokineResponsePredictor
    cytokine_encoder: LabelEncoder
    cell_type_encoder: LabelEncoder


def train_cytokine_predictor(
    latent: np.ndarray,
    cytokine_labels,
    cell_type_labels,
    *,
    n_epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> PredictorArtifacts:
    cytokine_encoder = LabelEncoder().fit(cytokine_labels)
    cell_encoder = LabelEncoder().fit(cell_type_labels)

    cyto_targets = torch.tensor(cytokine_encoder.transform(cytokine_labels), dtype=torch.long)
    cell_targets = torch.tensor(cell_encoder.transform(cell_type_labels), dtype=torch.long)
    latent_tensor = torch.tensor(latent, dtype=torch.float32)

    model = CytokineResponsePredictor(
        n_latent=latent.shape[1],
        n_cytokines=len(cytokine_encoder.classes_),
        n_cell_types=len(cell_encoder.classes_),
    )
    dataset = TensorDataset(latent_tensor, cyto_targets, cell_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(n_epochs):
        for batch_latent, batch_cyto, batch_cell in loader:
            optimizer.zero_grad()
            logits = model(batch_latent)
            cyto_logits = logits[:, : model.n_cytokines]
            cell_logits = logits[:, model.n_cytokines :]
            loss = criterion(cyto_logits, batch_cyto) + criterion(cell_logits, batch_cell)
            loss.backward()
            optimizer.step()

    return PredictorArtifacts(model=model, cytokine_encoder=cytokine_encoder, cell_type_encoder=cell_encoder)


def predict(
    artifacts: PredictorArtifacts, latent: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return predicted cytokine condition and cell type labels."""
    artifacts.model.eval()
    with torch.no_grad():
        latent_tensor = torch.tensor(latent, dtype=torch.float32)
        logits = artifacts.model(latent_tensor)
        cyto_logits = logits[:, : artifacts.model.n_cytokines]
        cell_logits = logits[:, artifacts.model.n_cytokines :]
        cyto_pred = cyto_logits.argmax(dim=1).cpu().numpy()
        cell_pred = cell_logits.argmax(dim=1).cpu().numpy()
    return (
        artifacts.cytokine_encoder.inverse_transform(cyto_pred),
        artifacts.cell_type_encoder.inverse_transform(cell_pred),
    )
