from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        
        self.input_layer = nn.Sequential(
            nn.Linear(4 * n_track, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        self.layer1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        self.output_layer = nn.Linear(512, 2 * n_waypoints)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.
        """
        track_left = track_left / 15.0
        track_right = track_right / 15.0

        x = torch.cat([track_left, track_right], dim=1).flatten(start_dim=1)
        
        x = self.input_layer(x)
        x = x + self.layer1(x)  # Residual connection
        x = x + self.layer2(x)  # Residual connection
        
        x = self.output_layer(x)
        return x.view(x.size(0), self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)
        
        # Project 2D coordinates into d_model dimensions
        self.input_proj = nn.Linear(2, d_model)
        
        # Positional embedding for the 2 * n_track input points
        self.pos_embed = nn.Embedding(2 * n_track, d_model)
        
        # Transformer Decoder layer and module
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # Output projection back to 2D
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b = track_left.size(0)
        
        # Scale inputs roughly to [-1, 1] range
        track_left = track_left / 15.0
        track_right = track_right / 15.0
        
        # Concatenate track points: shape (b, 2 * n_track, 2)
        track_points = torch.cat([track_left, track_right], dim=1)
        
        # Project to d_model: shape (b, 2 * n_track, d_model)
        memory = self.input_proj(track_points)
        
        # Create position indices: shape (2 * n_track)
        pos = torch.arange(2 * self.n_track, device=track_points.device)
        pos_embeddings = self.pos_embed(pos)
        
        # Add positional embeddings to memory
        memory = memory + pos_embeddings.unsqueeze(0)
        
        # Create target queries: shape (b, n_waypoints, d_model)
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        
        # Cross-attend: queries (tgt) attend to input points (memory)
        out = self.transformer(tgt=queries, memory=memory)
        
        # Project output back to 2D coordinates: shape (b, n_waypoints, 2)
        preds = self.output_proj(out)
        
        return preds


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        **kwargs,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        def block(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
            )

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.res1 = block(32, 64, stride=2)
        self.downsample1 = nn.Sequential(nn.Conv2d(32, 64, 1, stride=2, bias=False), nn.BatchNorm2d(64))
        
        self.res2 = block(64, 128, stride=2)
        self.downsample2 = nn.Sequential(nn.Conv2d(64, 128, 1, stride=2, bias=False), nn.BatchNorm2d(128))
        
        self.res3 = block(128, 256, stride=2)
        self.downsample3 = nn.Sequential(nn.Conv2d(128, 256, 1, stride=2, bias=False), nn.BatchNorm2d(256))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        x = self.conv1(x)
        x = nn.functional.relu(self.res1(x) + self.downsample1(x))
        x = nn.functional.relu(self.res2(x) + self.downsample2(x))
        x = nn.functional.relu(self.res3(x) + self.downsample3(x))
        
        x = self.classifier(x)

        return x.view(-1, self.n_waypoints, 2)



MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
