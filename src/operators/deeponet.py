"""
DeepONet and Graph Neural Operator (GNO) Architectures.

DeepONet: Learning operators via branch-trunk architecture.
GNO: Learning operators on graphs/meshes using message passing.

References:
- Lu et al., "Learning Nonlinear Operators via DeepONet" (Nature Machine Intelligence 2021)
- Li et al., "Neural Operator: Graph Neural Operators" (ICLR 2021)
- Brandstetter et al., "Message Passing Neural PDE Solvers" (ICLR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Callable
import math


class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet).
    
    Learns operators G: u -> G(u) using:
    - Branch net: encodes input function u at sensor points
    - Trunk net: encodes query points y
    - Dot product: G(u)(y) = sum_i b_i(u) * t_i(y)
    
    Supports both standard and stacked DeepONet variants.
    """
    
    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        output_dim: int = 1,
        branch_hidden: List[int] = [128, 128, 128],
        trunk_hidden: List[int] = [128, 128, 128],
        branch_layers: int = 3,
        trunk_layers: int = 3,
        p: int = 128,  # Latent dimension (number of basis functions)
        activation: Callable = F.gelu,
        branch_net: Optional[nn.Module] = None,
        trunk_net: Optional[nn.Module] = None,
        bias: bool = True,
        stacked: bool = False  # Stacked DeepONet
    ):
        super().__init__()
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.output_dim = output_dim
        self.p = p
        self.stacked = stacked
        
        # Branch network: maps input function to latent coefficients
        if branch_net is not None:
            self.branch = branch_net
        else:
            layers = []
            in_dim = branch_input_dim
            for h in branch_hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.LayerNorm(h))
                layers.append(activation)
                in_dim = h
            layers.append(nn.Linear(in_dim, p * output_dim if not stacked else p))
            self.branch = nn.Sequential(*layers)
        
        # Trunk network: maps query points to basis functions
        if trunk_net is not None:
            self.trunk = trunk_net
        else:
            layers = []
            in_dim = trunk_input_dim
            for h in trunk_hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.LayerNorm(h))
                layers.append(activation)
                in_dim = h
            layers.append(nn.Linear(in_dim, p * output_dim if not stacked else p))
            self.trunk = nn.Sequential(*layers)
        
        # Output bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter('bias', None)
    
    def forward(
        self, 
        u: torch.Tensor,  # (batch, branch_input_dim) or (batch, n_sensors, branch_input_dim)
        y: torch.Tensor   # (batch, n_query, trunk_input_dim) or (n_query, trunk_input_dim)
    ) -> torch.Tensor:
        """
        Args:
            u: Input function values at sensor points
            y: Query points where output is evaluated
        
        Returns:
            Output of shape (batch, n_query, output_dim)
        """
        # Branch output: (batch, p * output_dim) or (batch, n_sensors, p)
        b = self.branch(u)
        
        # Trunk output: (n_query, p * output_dim) or (batch, n_query, p)
        t = self.trunk(y)
        
        if self.stacked:
            # Stacked DeepONet: b and t are (batch, p), output is dot product
            # For each output dimension, we have separate branch/trunk
            out = torch.einsum('bp,bq->bpq', b, t)  # This is simplified
            # Actually stacked DeepONet needs more careful implementation
            raise NotImplementedError("Stacked DeepONet requires special implementation")
        else:
            # Standard DeepONet
            batch_size = u.shape[0]
            
            # Reshape for dot product
            if b.dim() == 2:
                b = b.view(batch_size, self.p, self.output_dim)  # (batch, p, output_dim)
            elif b.dim() == 3:
                b = b.view(batch_size, -1, self.p, self.output_dim).mean(1)  # Average over sensors
            
            if t.dim() == 2:
                t = t.view(-1, self.p, self.output_dim)  # (n_query, p, output_dim)
                t = t.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch, n_query, p, output_dim)
            elif t.dim() == 3:
                t = t.view(batch_size, -1, self.p, self.output_dim)
            
            # Dot product over p: (batch, n_query, output_dim)
            out = torch.einsum('bpo,bqpo->bqo', b, t)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class StackedDeepONet(nn.Module):
    """
    Stacked DeepONet for multi-output operators.
    
    Each output dimension has its own branch-trunk pair,
    allowing different basis functions for different outputs.
    """
    
    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        output_dim: int,
        p: int = 128,
        branch_hidden: List[int] = [128, 128, 128],
        trunk_hidden: List[int] = [128, 128, 128],
        activation: Callable = F.gelu,
        shared_branch: bool = False,
        shared_trunk: bool = False
    ):
        super().__init__()
        self.output_dim = output_dim
        self.p = p
        self.shared_branch = shared_branch
        self.shared_trunk = shared_trunk
        
        if shared_branch:
            self.branch = self._make_mlp(branch_input_dim, p, branch_hidden, activation)
        else:
            self.branches = nn.ModuleList([
                self._make_mlp(branch_input_dim, p, branch_hidden, activation)
                for _ in range(output_dim)
            ])
        
        if shared_trunk:
            self.trunk = self._make_mlp(trunk_input_dim, p, trunk_hidden, activation)
        else:
            self.trunks = nn.ModuleList([
                self._make_mlp(trunk_input_dim, p, trunk_hidden, activation)
                for _ in range(output_dim)
            ])
        
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def _make_mlp(self, in_dim: int, out_dim: int, hidden: List[int], act: Callable) -> nn.Sequential:
        layers = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(act)
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)
    
    def forward(
        self, 
        u: torch.Tensor,  # (batch, branch_input_dim)
        y: torch.Tensor   # (batch, n_query, trunk_input_dim)
    ) -> torch.Tensor:
        batch_size, n_query = u.shape[0], y.shape[1]
        
        outputs = []
        
        for i in range(self.output_dim):
            branch = self.branch if self.shared_branch else self.branches[i]
            trunk = self.trunk if self.shared_trunk else self.trunks[i]
            
            b = branch(u)  # (batch, p)
            t = trunk(y.view(-1, y.shape[-1])).view(batch_size, n_query, self.p)  # (batch, n_query, p)
            
            # Dot product
            out = torch.einsum('bp,bqp->bq', b, t)  # (batch, n_query)
            outputs.append(out)
        
        out = torch.stack(outputs, dim=-1)  # (batch, n_query, output_dim)
        out = out + self.bias
        
        return out


class PODDeepONet(nn.Module):
    """
    POD-DeepONet: Uses Proper Orthogonal Decomposition for trunk initialization.
    
    Pre-computes POD modes from training data and uses them to initialize
    the trunk network, improving data efficiency.
    
    Reference: "POD-DeepONet: A Deep Learning Framework for PDEs" (2022)
    """
    
    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        output_dim: int,
        pod_modes: torch.Tensor,  # (n_modes, n_query_points)
        p: int = 128,
        branch_hidden: List[int] = [128, 128, 128],
        trunk_hidden: List[int] = [128, 128, 128],
        activation: Callable = F.gelu,
        trainable_trunk: bool = True
    ):
        super().__init__()
        self.p = p
        n_modes = pod_modes.shape[0]
        
        # Branch network
        self.branch = self._make_branch(branch_input_dim, p * output_dim, branch_hidden, activation)
        
        # Trunk: initialized with POD modes
        self.pod_modes = nn.Parameter(pod_modes.float(), requires_grad=trainable_trunk)
        
        # Optional additional trunk layers
        if trunk_hidden:
            trunk_layers = []
            in_dim = trunk_input_dim
            for h in trunk_hidden:
                trunk_layers.append(nn.Linear(in_dim, h))
                trunk_layers.append(nn.LayerNorm(h))
                trunk_layers.append(activation)
                in_dim = h
            trunk_layers.append(nn.Linear(in_dim, n_modes))
            self.trunk_net = nn.Sequential(*trunk_layers)
        else:
            self.trunk_net = None
        
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def _make_branch(self, in_dim, out_dim, hidden, act):
        layers = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(act)
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)
    
    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = u.shape[0]
        n_query = y.shape[1]
        
        # Branch output
        b = self.branch(u).view(batch_size, self.p, self.output_dim)
        
        # Trunk output
        if self.trunk_net is not None:
            t = self.trunk_net(y.view(-1, y.shape[-1]))  # (batch * n_query, n_modes)
            t = t.view(batch_size, n_query, -1)
        else:
            t = self.pod_modes.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_query, n_modes)
        
        # Ensure compatible dimensions
        if t.shape[-1] != self.p:
            # Project or pad
            if t.shape[-1] < self.p:
                padding = torch.zeros(batch_size, n_query, self.p - t.shape[-1], device=t.device)
                t = torch.cat([t, padding], dim=-1)
            else:
                t = t[:, :, :self.p]
        
        t = t.unsqueeze(-1).expand(-1, -1, -1, self.output_dim)  # (batch, n_query, p, output_dim)
        
        # Dot product
        out = torch.einsum('bpo,bqpo->bqo', b, t)
        
        return out + self.bias


class GNOBlock(nn.Module):
    """
    Graph Neural Operator Block.
    
    Message passing on graph with spectral convolution in latent space.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        edge_attr_dim: int = 0,
        activation: Callable = F.gelu,
        norm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Message passing
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_attr_dim, out_channels),
            nn.LayerNorm(out_channels),
            activation,
            nn.Linear(out_channels, out_channels)
        )
        
        # Spectral convolution (global)
        self.spectral_conv = nn.Linear(modes, modes, dtype=torch.cfloat)
        
        # Update
        self.update_mlp = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.LayerNorm(out_channels),
            activation,
            nn.Linear(out_channels, out_channels)
        )
        
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = activation
    
    def forward(
        self, 
        x: torch.Tensor,           # (n_nodes, in_channels)
        edge_index: torch.Tensor,  # (2, n_edges)
        edge_attr: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None  # (n_nodes, dim) for spectral basis
    ) -> torch.Tensor:
        """
        Args:
            x: Node features
            edge_index: Graph connectivity
            edge_attr: Edge attributes
            pos: Node positions for spectral basis
        """
        n_nodes = x.shape[0]
        
        # Local message passing
        row, col = edge_index
        x_j = x[row]  # Source nodes
        x_i = x[col]  # Target nodes
        
        if edge_attr is not None:
            msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            msg_input = torch.cat([x_i, x_j], dim=-1)
        
        messages = self.msg_mlp(msg_input)
        
        # Aggregate
        aggr = torch.zeros_like(x)
        aggr.index_add_(0, col, messages)
        
        # Spectral convolution (global)
        if pos is not None:
            # Project to spectral basis
            # This is simplified - real implementation uses eigenvectors of Laplacian
            x_ft = torch.fft.rfft(x, dim=0)[:, :self.modes]
            x_ft = self.spectral_conv(x_ft)
            x_spec = torch.fft.irfft(x_ft, n=n_nodes, dim=0)
            
            # Combine local and global
            combined = torch.cat([aggr, x_spec], dim=-1)
        else:
            combined = torch.cat([aggr, x], dim=-1)
        
        # Update
        out = self.update_mlp(combined)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out


class GNO(nn.Module):
    """
    Graph Neural Operator for learning operators on unstructured meshes.
    
    Uses message passing with spectral convolution for global interactions.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 4,
        modes: int = 16,
        edge_attr_dim: int = 0,
        activation: Callable = F.gelu,
        norm: bool = True,
        dropout: float = 0.0,
        lifting_channels: int = 128,
        projection_channels: int = 128
    ):
        super().__init__()
        
        # Lifting
        self.lifting = nn.Sequential(
            nn.Linear(in_channels, lifting_channels),
            activation,
            nn.Linear(lifting_channels, hidden_channels)
        )
        
        # GNO blocks
        self.blocks = nn.ModuleList([
            GNOBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                modes=modes,
                edge_attr_dim=edge_attr_dim,
                activation=activation,
                norm=norm,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, projection_channels),
            activation,
            nn.Linear(projection_channels, out_channels)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (n_nodes, in_channels)
            edge_index: Graph edges (2, n_edges)
            edge_attr: Edge features (n_edges, edge_attr_dim)
            pos: Node positions (n_nodes, dim)
            batch: Batch assignment for batched graphs
        """
        # Lifting
        x = self.lifting(x)
        
        # GNO blocks
        for block in self.blocks:
            x = block(x, edge_index, edge_attr, pos) + x  # Residual
        
        # Projection
        x = self.projection(x)
        
        return x


class MNOBlock(nn.Module):
    """
    Multipole Neural Operator Block.
    
    Uses multipole expansion for efficient long-range interactions
    on graphs/point clouds.
    
    Reference: "Multipole Graph Neural Operator for PDEs" (2022)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_multipole: int = 16,
        levels: int = 3,
        **kwargs
    ):
        super().__init__()
        self.n_multipole = n_multipole
        self.levels = levels
        
        # Local message passing
        self.local_conv = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Multipole expansion
        self.multipole = nn.Parameter(
            torch.randn(n_multipole, n_multipole, dtype=torch.cfloat) * 0.1
        )
        
        # Combine
        self.combine = nn.Linear(out_channels * 2, out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor
    ) -> torch.Tensor:
        # Local interactions
        row, col = edge_index
        x_j = x[row]
        x_i = x[col]
        local = self.local_conv(torch.cat([x_i, x_j], dim=-1))
        
        # Aggregate local
        n_nodes = x.shape[0]
        local_aggr = torch.zeros(n_nodes, local.shape[-1], device=x.device)
        local_aggr.index_add_(0, col, local)
        
        # Multipole (simplified - real implementation uses hierarchical decomposition)
        # For now, use spectral as proxy
        x_ft = torch.fft.rfft(x, dim=0)[:, :self.n_multipole]
        x_ft = torch.einsum('nm,nmc->nc', x_ft, self.multipole)
        multipole_out = torch.fft.irfft(x_ft, n=n_nodes, dim=0)
        
        # Combine
        out = self.combine(torch.cat([local_aggr, multipole_out], dim=-1))
        
        return out


class MNO(nn.Module):
    """Multipole Neural Operator."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 4,
        n_multipole: int = 16,
        **kwargs
    ):
        super().__init__()
        
        self.lifting = nn.Linear(in_channels, hidden_channels)
        
        self.blocks = nn.ModuleList([
            MNOBlock(hidden_channels, hidden_channels, n_multipole)
            for _ in range(n_layers)
        ])
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Linear(hidden_channels * 2, out_channels)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor
    ) -> torch.Tensor:
        x = self.lifting(x)
        
        for block in self.blocks:
            x = block(x, edge_index, pos) + x
        
        x = self.projection(x)
        return x


class LNOBlock(nn.Module):
    """
    Low-Rank Neural Operator Block.
    
    Uses low-rank approximation of integral kernel for efficiency.
    
    Reference: "Low-Rank Neural Operators" (2023)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        n_points: int,
        **kwargs
    ):
        super().__init__()
        self.rank = rank
        self.n_points = n_points
        
        # Low-rank kernel: K(x,y) ≈ sum_r U_r(x) V_r(y)
        self.U = nn.Parameter(torch.randn(n_points, rank, in_channels) * 0.1)
        self.V = nn.Parameter(torch.randn(n_points, rank, out_channels) * 0.1)
        
        # Local
        self.local = nn.Conv1d(in_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_points, in_channels)
        """
        batch, n, c = x.shape
        
        # Low-rank integral: ∫ K(x,y) u(y) dy
        # K(x,y) ≈ U(x)^T V(y)
        # Output = U(x) @ (V(y)^T @ u(y))
        
        # V^T @ u: (batch, rank, out_channels)
        Vu = torch.einsum('bni,nro->bro', x, self.V)
        
        # U @ Vu: (batch, n_points, out_channels)
        integral = torch.einsum('nri,bro->bno', self.U, Vu)
        
        # Local
        local = self.local(x.transpose(1, 2)).transpose(1, 2)
        
        return integral + local


class LNO(nn.Module):
    """Low-Rank Neural Operator for 1D/2D problems."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 4,
        rank: int = 32,
        n_points: int = 1024,
        **kwargs
    ):
        super().__init__()
        
        self.lifting = nn.Conv1d(in_channels, hidden_channels, 1)
        
        self.blocks = nn.ModuleList([
            LNOBlock(hidden_channels, hidden_channels, rank, n_points)
            for _ in range(n_layers)
        ])
        
        self.projection = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels * 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_channels * 2, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, n_points)
        x = self.lifting(x)
        
        for block in self.blocks:
            x_t = x.transpose(1, 2)  # (batch, n_points, channels)
            out_t = block(x_t)
            x = out_t.transpose(1, 2) + x
        
        x = self.projection(x)
        return x


class FFNOBlock(nn.Module):
    """
    Fast Fourier Neural Operator Block.
    
    Uses factorized FFT for O(N log N) complexity instead of O(N^2).
    
    Reference: "Fast Fourier Neural Operators" (2023)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        rank: int = 0,
        **kwargs
    ):
        super().__init__()
        self.modes = modes
        self.rank = rank
        
        # Factorized spectral conv
        self.weights_x = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat) * 0.1
        )
        self.weights_y = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat) * 0.1
        )
        
        if rank > 0:
            self.U = nn.Parameter(torch.randn(in_channels, rank) * 0.1)
            self.V = nn.Parameter(torch.randn(rank, out_channels) * 0.1)
        
        self.local = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, c, h, w = x.shape
        
        # FFT x
        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')
        
        # Apply x weights
        m = min(self.modes, w // 2 + 1)
        x_ft[:, :, :, :m] = torch.einsum(
            'bcxy,ocy->bcxy', x_ft[:, :, :, :m], self.weights_x[:, :, :m]
        )
        
        # FFT y
        x_ft = torch.fft.rfft(x_ft, dim=-2, norm='ortho')
        
        # Apply y weights
        m = min(self.modes, h // 2 + 1)
        x_ft[:, :, :m, :] = torch.einsum(
            'bcxy,ocx->bcxy', x_ft[:, :, :m, :], self.weights_y[:, :, :m]
        )
        
        # Inverse FFT
        x = torch.fft.irfft(x_ft, dim=-2, norm='ortho')
        x = torch.fft.irfft(x, n=w, dim=-1, norm='ortho')
        
        # Low-rank correction
        if self.rank > 0:
            x_t = x.permute(0, 2, 3, 1)  # (b, h, w, c)
            lr = torch.einsum('bhwc,cr,ro->bhwo', x_t, self.U, self.V)
            x = x + lr.permute(0, 3, 1, 2)
        
        # Local
        x = x + self.local(x)
        
        x = self.norm(x)
        x = self.activation(x)
        
        return x


class FFNO(nn.Module):
    """Fast Fourier Neural Operator."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        n_layers: int = 4,
        modes: int = 16,
        rank: int = 0,
        **kwargs
    ):
        super().__init__()
        
        self.lifting = nn.Conv2d(in_channels, width, 1)
        
        self.blocks = nn.ModuleList([
            FFNOBlock(width, width, modes, rank)
            for _ in range(n_layers)
        ])
        
        self.projection = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifting(x)
        
        for block in self.blocks:
            x = block(x) + x
        
        x = self.projection(x)
        return x


# Factory
def get_deeponet(model_type: str, **kwargs) -> nn.Module:
    """Factory for DeepONet variants."""
    model_type = model_type.lower()
    if model_type == 'deeponet':
        return DeepONet(**kwargs)
    elif model_type == 'stacked_deeponet':
        return StackedDeepONet(**kwargs)
    elif model_type == 'pod_deeponet':
        return PODDeepONet(**kwargs)
    elif model_type == 'gno':
        return GNO(**kwargs)
    elif model_type == 'mno':
        return MNO(**kwargs)
    elif model_type == 'lno':
        return LNO(**kwargs)
    elif model_type == 'ffno':
        return FFNO(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    'DeepONet', 'StackedDeepONet', 'PODDeepONet',
    'GNO', 'GNOBlock',
    'MNO', 'MNOBlock',
    'LNO', 'LNOBlock',
    'FFNO', 'FFNOBlock',
    'get_deeponet',
]