import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PINN(nn.Module):
    """
    Continuous approximator u_theta(t,x).
    Inputs: [t, x_1, x_2]
    Output: u (tumor density)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, t, x1, x2):
        inputs = torch.cat([t, x1, x2], dim=1)
        return self.net(inputs)

def pde_residual_loss(model, t, x1, x2, D, rho, K):
    """
    Calculates the PDE residual loss: R_theta(t,x) = ∂u/∂t - D∇²u - ρu(1-u/K) ≈ 0
    Assumes no radiation during training for simplicity (or can be added).
    """
    t.requires_grad = True
    x1.requires_grad = True
    x2.requires_grad = True
    
    u = model(t, x1, x2)
    
    # Gradients
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    
    u_x1 = torch.autograd.grad(u.sum(), x1, create_graph=True)[0]
    u_x1x1 = torch.autograd.grad(u_x1.sum(), x1, create_graph=True)[0]
    
    u_x2 = torch.autograd.grad(u.sum(), x2, create_graph=True)[0]
    u_x2x2 = torch.autograd.grad(u_x2.sum(), x2, create_graph=True)[0]
    
    laplacian = u_x1x1 + u_x2x2
    
    # Physics equation residual
    residual = u_t - (D * laplacian) - (rho * u * (1.0 - u / K))
    
    return torch.mean(residual ** 2)

def train_pinn(epochs: int = 1000):
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Physics parameters
    D = 0.05
    rho = 0.1
    K = 1.0
    
    print("Starting PINN offline training...")
    
    for epoch in range(epochs):
        # Sample collocation points
        t_colloc = torch.rand((1000, 1)) * 20.0
        x1_colloc = torch.rand((1000, 1)) * 10.0
        x2_colloc = torch.rand((1000, 1)) * 10.0
        
        optimizer.zero_grad()
        loss = pde_residual_loss(model, t_colloc, x1_colloc, x2_colloc, D, rho, K)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Residual Loss: {loss.item():.6f}")
            
    print("PINN training complete. Ready to generate rollouts for SuperNet.")
    return model

if __name__ == "__main__":
    train_pinn(100)
