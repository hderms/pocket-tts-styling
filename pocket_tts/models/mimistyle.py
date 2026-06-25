import logging

from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.stateful_module import init_states
import torch
from torch import nn
import torch.nn.functional as F

from pocket_tts.modules.conv import pad_for_conv1d
from pocket_tts.models.melspectrogramloss import MelSpectrogramLoss

logger = logging.getLogger()

class PredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int , output_dim: int ):
        """
        A standard Prediction Head architecture.
        Typically features a bottleneck design with BatchNorm.
        """
        super().__init__()
        
        self.head = nn.Sequential(
            # 1. Expand or compress to a hidden dimension
            nn.Linear(input_dim, hidden_dim, bias=False),
            
            # 2. Normalize to stabilize the prediction representations
            nn.BatchNorm1d(hidden_dim),
            
            # 3. Non-linearity
            nn.ReLU(),
            
            # 4. Final linear projection to match the target conditioning vector
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.mean(dim=2))



import torch
import torch.nn as nn

class MimiMLPConditioner(nn.Module):
    def __init__(self, control_dim: int, latent_dim: int = 32, hidden_dim: int = 128):
        """
        A simple MLP projection layer for additive conditioning in the Mimi bottleneck.
        
        Args:
            control_dim: The dimension of your input control vector.
            latent_dim: The continuous latent dimension of Mimi (default is 32).
            hidden_dim: The hidden layer size of the MLP.
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(control_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Zero-initialization is critical here.
        # By initializing the final weights and biases to exactly 0, the MLP
        # will output a vector of pure zeros at the start of training.
        # This ensures the operation starts as a perfect identity function (z + 0 = z),
        # preventing the frozen decoder from collapsing due to chaotic noise.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Continuous latents from the frozen Mimi encoder. Shape: [Batch, latent_dim, Time]
            c: The conditioning control vector. Shape: [Batch, control_dim]
            
        Returns:
            Conditioned latents. Shape: [Batch, latent_dim, Time]
        """
        # 1. Project the control vector to match the latent channel dimension
        # shift shape: [Batch, latent_dim]
        shift = self.mlp(c)
        
        # 2. Reshape to allow broadcasting across the Time dimension
        # shift shape becomes: [Batch, latent_dim, 1]
        shift = shift.unsqueeze(-1)
        
        # 3. Apply the additive conditioning
        # The shift is broadcast and added to every frame in the sequence
        z_conditioned = z + shift

        

        return  z_conditioned
class MimiFiLMConditioner(nn.Module):
    def __init__(self, control_dim: int, latent_dim: int = 32, hidden_dim: int = 128):
        """
        A FiLM layer designed to modulate the 32-dimensional continuous Mimi latents.
        
        Args:
            control_dim: The size of your input control vector (e.g., 256 for a speaker embedding).
            latent_dim: The channel dimension of the Mimi codec bottleneck (default: 32).
            hidden_dim: The size of the hidden layer in the MLP.
        """
        super().__init__()
        
        # An MLP that projects the control vector into 2 * latent_dim
        # We need two values (gamma and beta) for every latent channel.
        self.mlp = nn.Sequential(
            nn.Linear(control_dim, hidden_dim),
            nn.SiLU(), # SiLU/Swish is standard for modern continuous audio models
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # Initialize the final layer to zero. 
        # This ensures that at the start of training, gamma=0 and beta=0,
        # meaning the FiLM layer acts as a perfect identity function: z_new = z
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: The continuous latents from Mimi. Shape: [Batch, latent_dim, Time]
            c: The conditioning control vector. Shape: [Batch, control_dim]
            
        Returns:
            Conditioned latents of the same shape as z: [Batch, latent_dim, Time]
        """
        # 1. Project control vector
        # film_params shape: [Batch, latent_dim * 2]
        film_params = self.mlp(c)
        
        # 2. Split into scale (gamma) and shift (beta)
        # Both shapes: [Batch, latent_dim]
        gamma, beta = film_params.chunk(2, dim=-1)
        
        # 3. Reshape for broadcasting across the Time dimension
        # Shapes become: [Batch, latent_dim, 1]
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        
        # 4. Apply modulation
        # We use (1 + gamma) so that when gamma is initialized to 0, 
        # the multiplier is 1 (Identity operation).
        z_conditioned = z * (1 + gamma) + beta
        
        return z_conditioned

class MimiStyleModel(nn.Module):
    def __init__(
        self,
        mimicodec: MimiModel,
        control_dim: int,
        latent_dim: int,
        batch_size: int,
        hidden_dim: int,
        device: torch.device,
        debug: bool = False,
    ):
        super().__init__()
        self.conditioning_mlp_layer = MimiMLPConditioner(control_dim=control_dim, latent_dim = latent_dim, hidden_dim = hidden_dim).to(device)
        self.mimi = mimicodec.to(device)

        self.prediction_head = PredictionHead(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=control_dim).to(device)
       
        self.batch_size = batch_size
        self.debug = debug

        # Stateful modules require cache/offset initialization before decoding.
        # We initialize the state dictionary with a sequence length buffer.
        self.device = device
        self.mel_loss = MelSpectrogramLoss(self.mimi.sample_rate)

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate() / self.frame_rate())

    def sample_rate(self) -> int:
        return self.mimi.sample_rate

    def frame_rate(self) -> float:
        return self.mimi.frame_rate

    def _to_framerate(self, x: torch.Tensor):
        return self.mimi._to_framerate(x)
      

    def _to_encoder_framerate(self, x: torch.Tensor, mimi_state) -> torch.Tensor:
        return self.mimi._to_encoder_framerate(x, mimi_state)

    def forward(self, x: torch.Tensor, c: torch.Tensor):

        self.mimi_state = init_states(self.mimi, batch_size=self.batch_size, sequence_length=10000)
        latents = None
        with torch.no_grad():
        # =====================================================================
        # STAGE 1: AUDIO -> ENCODER -> LATENTS
        # =====================================================================
        # Uses the method defined in MimiModel to pad and process through SEANet
            latents = self.encode_to_latent(x)

        # =====================================================================
        # STAGE 2: LATENTS -> CONDITIONED LATENTS
        # =====================================================================
        conditioned_latents = self.conditioning_mlp_layer.forward(latents.to(self.device), c)

        pred = self.prediction_head.forward(conditioned_latents.to(self.device))
        pred_loss = 1 - F.cosine_similarity(pred, c, dim=1).mean()

        # =====================================================================
        # STAGE 3: CONDITIONED LATENTS -> DECODER -> AUDIO
        # =====================================================================


        
        
        # Mirroring the internal logic of TTSModel._decode_and_dump:
        # Pass through the DummyQuantizer if dimensions align, otherwise pass raw.
        if latents.shape[1] == self.mimi.quantizer.dimension:
            latent_to_decode = self.mimi.quantizer(conditioned_latents)
        else:
            latent_to_decode = conditioned_latents
        if self.debug:
            print("prediction = ", pred)
            print("c = ", c)


 

        
        with torch.no_grad():
        # Decode back to waveform using the instantiated state
            reconstructed_wav = self.decode_from_latent(latent_to_decode, self.mimi_state)
        mel_loss = self.mel_loss.forward(reconstructed_wav, x)

        loss = mel_loss.mean() + pred_loss
        return loss, reconstructed_wav


    def decode_from_latent(self, latent: torch.Tensor, mimi_state) -> torch.Tensor:
        return self.mimi.decode_from_latent(latent, mimi_state)

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.mimi.encode_to_latent(x)


