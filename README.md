# Agents in JAX (AJAX): A JAX-Based Library for Modular and Efficient RL Agents

AJAX is a high-performance reinforcement learning library built entirely on **JAX**. It provides a modular and extensible framework for implementing and training RL agents, enabling **massive speedups** for parallel experiments on **TPUs/GPUs**. AJAX is designed to be **efficient**, **scalable**, and **easy to modify**, making it ideal for both research and production use.

---

## 🚀 Features

### **Feature Comparison**

| **Features**                          | **AJAX**          |
| ------------------------------------- | ----------------- |
| End-to-End JAX Implementation         | :heavy_check_mark: |
| Modular Design                        | :heavy_check_mark: |
| GPU/TPU Acceleration                  | :heavy_check_mark: |
| Logging Support                       | :heavy_check_mark: |
| Weights & Biases (wandb) Integration  | :heavy_check_mark: |
| Termination/Truncation handling       | :heavy_check_mark: |
| Documentation                         | :soon:             |
| Recurrent Network Support             | :soon:             |


---

### **End-to-End JAX**
- Fully implemented in JAX, ensuring seamless integration and hardware acceleration.
- Enables efficient parallelization for large-scale experiments on GPUs/TPUs.

### **Modular Design**
- Easily customize networks, optimizers, and training loops.
- Designed for researchers to quickly prototype and modify agents.

### **Available Agents**
- **Soft Actor-Critic (SAC)**: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al, 2018](https://arxiv.org/abs/1801.01290)
- **Average-reward Soft Actor-Critic (ASAC)**: Average-reward version of SAC. [Average-Reward Soft Actor-Critic, Adamczyk et al. 2025](https://arxiv.org/pdf/2501.09080v2). 
- **Proximal Policy Optimization (PPO)**: [Proximal Policy Optimization Algorithms, Schulman et al, 2017](https://arxiv.org/abs/1707.06347)
- **Average-Reward Policy Optimization (APO)**: [Average-Reward Reinforcement Learning with Trust Region Methods, Ma et al, 2021](https://arxiv.org/abs/2106.03442)
- **Action-Value Gradient (AVG)**: [Deep Policy Gradient Methods Without Batch Updates, Target Networks, or Replay Buffers, Vasan et al, 2024](https://arxiv.org/abs/2411.15370)
- **Randomized Ensembled Double Q-Learning (REDQ)**: [Randomized Ensembled Double Q-Learning: Learning Fast Without a Model, Chen et al, 2021](https://arxiv.org/abs/2101.05982)
- More agents to come!

### **Environment Compatibility**
- Works seamlessly with **Gymnax** and **Brax** environments, including the handling of truncation vs termination of episodes.
- Supports both single and parallel environments.

### **Replay Buffer**
- Efficient trajectory storage and sampling using **flashbax**.

### **Highly Optimized**
- Memory-efficient updates using `donate_argnums`.
- JAX-based implementation for GPU/TPU acceleration.

### **Parallel Logging**
- Allows for **live logging from multiple parallel run on a single device**, directly to Weight & Biases !

### **Upcoming Features**
- **Recurrent Support**: Optional LSTM-based recurrent networks for partially observable environments.
- **Profiling**: Profiling tool to cleanly measure the performance (speed and RAM) of jax agents.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YannBerthelot/Ajax.git
   cd ajax
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

---

## 📖 Usage

### **Training an Agent**
To train an agent using AJAX, run the following command:
```python
env_id = "halfcheetah"
SAC_agent = SAC(
        env_id=env_id,
    )
SAC_agent.train(seed=[1,2,3], n_timesteps=int(1e6))
```
Replace `<environment_name>` with the desired environment (e.g., `gymnax.CartPole-v1`).


---

## 📂 Project Structure

- **`ajax/`**: Core library containing implementations of agents and utilities.
- **`tests/`**: Unit tests for the framework.

---

## 🌟 Why AJAX?

1. **End-to-End JAX**: AJAX is fully implemented in JAX, allowing for unparalleled speed and efficiency in reinforcement learning workflows.
2. **Modular and Extensible**: Researchers can easily modify and extend the library to suit their needs.
3. **Scalable**: Designed to handle large-scale experiments with parallel environments on GPUs/TPUs.
4. **Future-Proof**: AJAX is continuously evolving, with more agents and features planned for future releases.
5. **Experiment Tracking**: Support for logging and wandb integration will make tracking experiments seamless.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a clear description of your changes.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use **TargetGym** in your research or project, please cite it as:

```bibtex
@misc{ajax2025,
  title        = {Ajax: Reinforcement Learning Agents in Jax},
  author       = {Yann Berthelot},
  year         = {2025},
  url          = {https://github.com/YannBerthelot/Ajax},
}
```
