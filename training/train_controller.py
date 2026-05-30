import torch
import torch.nn.functional as F
import numpy as np
import cma
import os

from wm.env import make_env
from wm.modeling import VAE, MDN_RNN, Controller

DEVICE = 'cpu' # Running environments heavily uses CPU, so keeping tensors on CPU is often faster to avoid data transfer bottlenecks
ENV_NAME = "VizdoomTakeCover-v1"
POPULATION_SIZE = 64
GENERATIONS = 50

def get_flat_weights(model):
    """Extract model weights to a 1D numpy array."""
    weights = []
    for param in model.parameters():
        weights.append(param.detach().cpu().numpy().flatten())
    return np.concatenate(weights)

def set_flat_weights(model, flat_weights):
    """Inject 1D numpy array weights back into the model."""
    offset = 0
    for param in model.parameters():
        shape = param.shape
        size = np.prod(shape)
        param.data.copy_(torch.tensor(flat_weights[offset:offset+size]).view(shape))
        offset += size

def evaluate_controller(flat_weights, vae, mdn, controller, env_name, action_dim, is_discrete):
    """Run one full episode and return the total reward."""
    # Load the candidate weights
    set_flat_weights(controller, flat_weights)
    
    env = make_env(env_name)
    obs, _ = env.reset()
    
    total_reward = 0
    done = False
    
    # Initialize the hidden state (h, c) for the LSTM
    # h shape: (1, 1, hidden_dim) since it's a single batch, single layer
    h = None 
    
    with torch.no_grad():
        while not done:
            # 1. Vision: Encode the observation
            # Convert obs to (1, C, H, W) float tensor
            obs_tensor = torch.tensor(obs * 255.0, dtype=torch.float32).to(DEVICE) / 255.0
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Get the 32D latent vector
            _, _, _, z = vae.forward_z(obs_tensor)
            
            # 2. Controller: Pick an action based on z and h
            # If h is None (first step), use zeros
            current_h = torch.zeros(1, mdn.hidden_dim).to(DEVICE) if h is None else h[0].squeeze(0).unsqueeze(0)
            
            action_logits = controller(z, current_h)
            
            if is_discrete:
                action = torch.argmax(action_logits, dim=-1).item()
                # Prepare one-hot action for the MDN-RNN
                action_tensor = F.one_hot(torch.tensor([[action]]), num_classes=action_dim).float().to(DEVICE)
            else:
                action = action_logits.squeeze(0).cpu().numpy()
                action_tensor = action_logits.unsqueeze(0).to(DEVICE)
                
            # Step the environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # 3. Memory: Update the hidden state for the NEXT step
            # Pass (batch=1, seq=1, dim) through MDN-RNN to step the LSTM
            z_seq = z.unsqueeze(0) # (1, 1, 32)
            _, _, _, h, _, _, _ = mdn(z_seq, action_tensor, h)
            
    env.close()
    return total_reward

if __name__ == '__main__':
    print("Setting up Controller Evolution...")
    
    # Environment info
    dummy_env = make_env(ENV_NAME)
    if hasattr(dummy_env.action_space, 'n'):
        ACTION_DIM = dummy_env.action_space.n
        IS_DISCRETE = True
    else:
        ACTION_DIM = dummy_env.action_space.shape[0]
        IS_DISCRETE = False
    dummy_env.close()

    # Load frozen VAE and MDN
    vae = VAE().to(DEVICE)
    vae.load_state_dict(torch.load("checkpoints/vae.pth", map_location=DEVICE, weights_only=True))
    vae.eval()
    
    mdn = MDN_RNN(action_dim=ACTION_DIM).to(DEVICE)
    # mdn.load_state_dict(torch.load("checkpoints/mdn.pth", map_location=DEVICE, weights_only=True))
    mdn.eval()
    
    # Initialize the tiny Controller
    controller = Controller(action_dim=ACTION_DIM, latent_dim=32, hidden_dim=256).to(DEVICE)
    
    # Extract starting weights as a flat array
    initial_weights = get_flat_weights(controller)
    num_params = len(initial_weights)
    print(f"Controller has {num_params} parameters.")
    
    # Initialize CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(initial_weights, 0.1, {'popsize': POPULATION_SIZE})
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for gen in range(GENERATIONS):
        # 1. Ask for candidate solutions
        solutions = es.ask()
        
        # 2. Evaluate all candidates in the environment
        # (In a real setup, you'd use Python multiprocessing here to run these in parallel!)
        rewards = []
        for i, weights in enumerate(solutions):
            reward = evaluate_controller(weights, vae, mdn, controller, ENV_NAME, ACTION_DIM, IS_DISCRETE)
            rewards.append(reward)
            
        # CMA-ES minimizes, so we pass negative rewards
        es.tell(solutions, [-r for r in rewards])
        
        best_reward = np.max(rewards)
        avg_reward = np.mean(rewards)
        print(f"Generation {gen+1} | Best Reward: {best_reward:.2f} | Avg Reward: {avg_reward:.2f}")
        
        # Save the best weights
        set_flat_weights(controller, es.result.xbest)
        torch.save(controller.state_dict(), "checkpoints/controller.pth")
        
    print("Evolution complete!")
