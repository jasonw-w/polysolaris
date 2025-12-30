import json
import os
import sys
import torch
import numpy as np

# Add src/ThreeBodyProblem to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# src/ThreeBodyProblem/RL_agent -> src/ThreeBodyProblem
three_body_dir = os.path.dirname(current_dir)
sys.path.append(three_body_dir)

from env.three_body_gym import ThreeBodyEnv
from RL_agent.model import Actor_Critic

def create_body(name, body_id, mass, position, velocity, colour, simulation=1, 
                create_stable_orbit=0, eccentricity=0, central_body_id=0):
    """
    Creates a dictionary representing a celestial body for the simulation.
    """
    if central_body_id == body_id:
         central_body_id = None

    return {
        "name": name,
        "id": body_id,
        "mass": mass,
        "initial_position": position,
        "initial_velocity": velocity,
        "create_stable_orbit": create_stable_orbit,
        "eccentricity": eccentricity,
        "id_of_central_body": central_body_id,
        "colour": colour,
        "simulation": simulation
    }

def main():
    # Define masses as in training or desired
    masses_val = [1, 1, 1]
    
    # Setup paths
    project_root = os.path.abspath(os.path.join(three_body_dir, "..", ".."))
    model_path = os.path.join(project_root, "best_agent.pth")
    output_path = os.path.join(project_root, "data", "predicted_system.json")
    
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print("Error: best_agent.pth not found.")
        return

    # Initialize environment
    # We use random positions/velocities for initialization, 
    # but the agent will predict the *correct* initial velocity for the *given* Position.
    # Note: If the agent was trained to find velocities for ANY position, we can pass random.
    # If it was trained on fixed positions, we should use those.
    # Based on train.py, it was initialized with randn(3,3). 
    # But does env.reset() re-randomize? 
    # We will assume we want to generate a valid trajectory for a *new* random configuration 
    # or the standard one. Let's use what env.reset() gives us.
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = ThreeBodyEnv(
        masses=torch.tensor(masses_val, dtype=torch.float32),
        initial_pos=torch.randn(3,3),
        initial_v=torch.randn(3,3)
    )
    
    # Initialize Agent
    num_input = env.observation_space.shape[0]
    num_output = env.action_space.shape[0]
    agent = Actor_Critic(num_input, num_output).to(device)
    
    # Load Weights
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval() # Set to evaluation mode
    
    # Get a state
    # env.reset() will return the initial state (positions, velocities, masses)
    # Note: The 'velocities' in the state are the initial (random/zero) ones. 
    # The agent's JOB is to look at this state (mainly positions/masses) and output NEW velocities.
    state, _ = env.reset()
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        # returns action_mean, value, log_std
        action_mean, _, _ = agent(state_t)
        # The action represents the predicted velocity
        predicted_velocities = action_mean.cpu().numpy()[0].reshape(3, 3)
        
    # Extract positions from the state (first 9 elements are flat positions)
    flat_pos = state[:9]
    positions = flat_pos.reshape(3, 3)
    
    # Generate JSON
    planets = []
    colors = ["yellow", "blue", "grey"]
    names = ["Star", "Planet1", "Planet2"]
    
    # Scale inputs?
    # The simulation/agent seems to use unitless or normalized values (randn). 
    # The visualization/loadplanets might expect these values or scaled ones.
    # Since `loadplanets` loads directly into `solar_sys_body`, the values in JSON 
    # are used as-is. If the agent outputs small values (randn range), and the 
    # visualizer expects AU (0-50) or pixels, it might be small.
    # But `train.py` used `randn`. 
    # However, `solar_system.json` uses masses like 1.9e7. 
    # The agent uses masses [100, 1, 0.01].
    # This implies the agent is trained on a DIFFERENT scale than `solar_system.json`.
    # If we generate `predicted_system.json` with mass=100 and pos ~ 1.0, 
    # it might look different or invisible if the camera is zoomed out for 10^7 mass systems.
    # BUT, the user asked to generate output *from the agent*. So we should use the agent's scale.
    # The user can zoom in or the visualization might auto-scale.
    
    for i in range(len(masses_val)):
        body = create_body(
            name=names[i],
            body_id=i,
            mass=masses_val[i],
            position=positions[i].tolist(),
            velocity=predicted_velocities[i].tolist(),
            colour=colors[i],
            simulation=1,
            create_stable_orbit=0, # IMPORTANT: Use agent's velocity!
            eccentricity=0,
            central_body_id=0 if i > 0 else None
        )
        planets.append(body)

    system_data = {
        "planets": planets
    }

    with open(output_path, 'w') as f:
        json.dump(system_data, f, indent=2)
    
    print(f"Successfully generated predicted system at: {output_path}")
    print("Sample Body 1:")
    print(f"  Pos: {positions[1]}")
    print(f"  Vel: {predicted_velocities[1]}")

if __name__ == "__main__":
    main()
