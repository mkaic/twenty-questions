import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from typing import Tuple

EPSILON = 1e-6

import math
import torch


def get_rotary_position_vectors(meshgrid_shape, dim, device):

    positions = torch.stack(
        torch.meshgrid(
            *[
                torch.arange(i, dtype=torch.float32, device=device)
                for i in meshgrid_shape
            ],
            indexing="ij"
        ),
        dim=-1,
    )

    freq_bands = []

    assert dim % 2 == 0, "dim must be even"
    num_frequencies = dim // 2

    for freq_idx in range(1, num_frequencies + 1):
        for pe_axis in range(len(meshgrid_shape)):
            pos = positions[..., pe_axis] * (
                1 / (10000 ** (freq_idx / (num_frequencies)))
            )
            freq_bands.append(torch.cos(pos))
            freq_bands.append(torch.sin(pos))

    positions = torch.stack(freq_bands, dim=-1)  # *meshgrid_shape, dim

    return positions

def hypersphere(x):
    return F.normalize(x, p=2, dim=-1) * math.sqrt(x.shape[-1])


class Questioner(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        num_layers: int,
        questions_per_layer: int,
        question_vector_size: int,
        context_vector_size: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.questions_per_layer = questions_per_layer
        self.question_vector_size = question_vector_size
        self.context_vector_size = context_vector_size
        self.hidden_dim = hidden_dim

        self.color_embedding = nn.Linear(input_channels, question_vector_size)
        self.initial_context = nn.Parameter(torch.randn(1, context_vector_size))
        self.initial_questions = nn.Parameter(
            torch.randn(questions_per_layer, question_vector_size)
        )

        self.brain = nn.Sequential(
            nn.Linear(
                self.questions_per_layer + self.context_vector_size, self.hidden_dim
            ),
            nn.GELU(),
            nn.Linear(
                self.hidden_dim,
                self.questions_per_layer * self.question_vector_size
                + self.questions_per_layer
                + self.context_vector_size,
            ),
        )

        self.output_head = nn.Sequential(
            nn.Linear(
                self.questions_per_layer + self.context_vector_size, self.hidden_dim
            ),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        self.rotary_positions_cache = None
        self.input_shape = None

    def forward(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:

        batch_size = tokens.shape[0]
        input_channels = tokens.shape[-1]

        tokens = tokens.view(batch_size, -1, input_channels)

        # tokens: (b, seq, input_channels)
        batch_size, sequence_length, input_channels = tokens.shape

        tokens = self.color_embedding(tokens)  # (b, seq, qdim)

        shape_is_none = self.input_shape is None
        shape_is_different = self.input_shape != (sequence_length, input_channels)
        if shape_is_none or shape_is_different:
            self.input_shape = (sequence_length, input_channels)
            self.rotary_positions_cache = get_rotary_position_vectors(
                meshgrid_shape=(sequence_length,),
                dim=self.question_vector_size,
                device=tokens.device,
            )  # (seq, qdim)

        posenc = self.rotary_positions_cache.unsqueeze(0)

        tokens = hypersphere(tokens)
        tokens = tokens * posenc

        questions = self.initial_questions
        answers = torch.matmul(
            questions,  # (qnum, qdim)
            tokens.unsqueeze(-1),  # (b, seq, qdim, 1)
        )
        # (b, seq, qnum, qdim) @ (b, seq, qdim, 1) -> (b, seq, qnum, 1)
        # answers = torch.sigmoid(answers)
        answers = torch.squeeze(answers, dim=-1)  # (b, seq, qnum)
        answers = torch.mean(answers, dim=1)  # (b, qnum)

        context = self.initial_context.expand(batch_size, -1)

        for i in range(self.num_layers):

            input_to_brain = torch.cat([context, answers], dim=-1)
            output_from_brain: torch.Tensor = self.brain(input_to_brain)

            questions = output_from_brain[
                ..., : self.question_vector_size * self.questions_per_layer
            ]
            questions = questions.reshape(
                batch_size,
                1,
                self.questions_per_layer,
                self.question_vector_size,
            )
            
            context = context + output_from_brain[..., -self.context_vector_size :]

            answers = torch.matmul(
                questions,  # (qnum, qdim)
                tokens.unsqueeze(-1),  # (b, seq, qdim, 1)
            )
            # (b, seq, qnum, qdim) @ (b, seq, qdim, 1) -> (b, seq, qnum, 1)
            answers = torch.squeeze(answers, dim=-1)  # (b, seq, qnum)
            answers = torch.mean(answers, dim=1)  # (b, qnum)

        output = self.output_head(torch.cat([context, answers], dim=-1))

        return output
