# Autonomus Driving with Deep Reinforcement Learning
This project is a deep reinforcement learning project to demonstrate more reinforcement learning skills, as well as tackle a problem I am interested in solving - that being autonomous driving,

## Overview
While one of the goals of this project is to demonstrate knowledge in reinforcement learning, I am also interested in and curious in seeing how we can use deeper networks to tackle autonomus driving. To start, we can go over the learning algorithm that I intend to use throughout this project, with other algorithms being tested to see which performs best.

## Proximal Policy Optimization
**Proximal policy optimiation** (PPO) is a reinforcement learning algorithm that essentially allows for us to be able to work with continuous action spaces. We can think about how when using a deep q-network or a double q-network that the goal of those algorithms is to train an RL agent on a discrete action space. In Laymen's terms - with a discrete action space, we're working in an action space that has a countable number of actions. However, working in continuous action space, we can have infinitely many actions.

### Continuous Action Spaces
To address how autonomous driving is a **continuous action space**, we can think about how much pressure we should give the throttle when accelerating. With this example - we can define a domain for throttle pressure, d, where $d = [0, 1]$, where 0 is applying no pressure and 1 is full-throttling. If we wanted to move the car, we would apply a small amount of pressure. The amount we step on the gas can be represented as a value within our domain of throttle pressure. If we want to accelerate slowly, we can apply light throttle pressure - say 0.25, or 25% of the throttle pressed down. 

Now, that may appear quantifiable. We can discretize the action space and say we have 100 possible throttle inputs, {0, 0.01, 0.02, ..., 0.99, 1.0}. However, how would we handle the event where we don't want to apply exactly 49% or 50% of throttle pressure? But what if the optimal amount of throttle pressure we need is 0.494 or 0.4944? As our agent tries to get increasingly more precise with throttle input, the action value the agent can decide to take becomes infinitesimally smaller. Therefore, to achieve fine-grained control, our action space must be continous on the domain d, $d = [0, 1]$.

### Handling Continuous Action Spaces
To handle continuous action spaces, we can use [PPO](https://arxiv.org/pdf/1707.06347) - first published by OpenAI in 2017. To explain it more intuitively before diving into the math of the algorithm, PPO almost works exactly like a Deep Q-Network (DQN). In a DQN, we have two neural networks, one that approximates the q-value function - which we know is the network approximating the Bellman Equation: $$Q(s_t, a_t) = \mathbb{E}[r_t + \gamma (\max_{a'} Q(s_{t+1}, a'))]$$ 

With PPO, instead of approxmating the best state-action value-pair with max reward, we try to compute the maximum on-policy reward, $\pi_\theta(a|s)$.

## References

- Schulman, J., et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)