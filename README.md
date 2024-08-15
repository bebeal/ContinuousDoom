# ContinuousDoom

Learning to play Doom with deep reinforcement learning using a continuous action.

This agent plays the game using similar input devices as humans play the game with, a keyboard (categorical distribution) and mouse (continuous distribution).

* [Downloadable Weights](https://drive.google.com/file/d/1XvftbgcGp8dwgWLwqJRd8W2ftGFaVzjc/view?usp=sharing)
* [Colab Script using trained agent](https://colab.research.google.com/drive/1VcGCq4awrS8Icl8xTWKS7qb0WsNTkpZB?usp=sharing)

Scale this to modern Doom VR and pair it with a VR headset and I think this turns into a money printer backed by an alphabet

## The Environment

I'll be using [VizDoom](https://arxiv.org/pdf/1809.03470.pdf) which is essentially a contained little Doom emulator that allows for customization and easy control. It includes a multitude of scenarios that have particular tasks and goals for your agent to solve. I made a (overly complicated) gym wrapper to simplify it's internal parameter configuration. Also available here if interested [VizDoomGym](https://github.com/bebeal/VizDoomGym)

### Defend The Center

The purpose of this scenario is to teach the agent that killing the monsters is GOOD and when monsters kill you is BAD. In addition, wasting ammunition is not very good either. Agent is rewarded only for killing monsters so they have to figure out the rest for themselves.

Map is a large circle. Player is spawned in the exact center. 5 melee-only, monsters are spawned along the wall. Monsters are killed after a single shot. After dying each monster is respawned after some time. Episode ends when the player dies (it's inevitable because of limited ammo).

* +1 for killing a monster
* -1 for dying

Additionally, I added some configuration within my wrapper to end the scenario if the agent is out of ammo for 4 steps in a row.

## The Agent

The foundation of my agent relies on a PPO algorithm with a variety of tweaks/changes. PPO stands for [Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347.pdf) and it is a core RL algorithm that has produced promising results in many areas of RL. However, the original PPO paper conveniently left out a ton of code-level optimizations that are arguably necessary to make it work in practice. Luckily these optimizations were studied and exposed by another paper ["Implementation Matters In Deep Policy Gradients: A Case Study On PPO And TRPO"](https://arxiv.org/pdf/2005.12729.pdf) and are used here. 

Additionally, we are using an actor-critic style agent where the actor and the critic have a separate loss. We will also add an additional entropy loss term to encourage exploration. Lastly, PPO algorithms typically parallelize their agents to collect lots of experience fast, and run multiple backprop updates (typically known as "k epochs") using the same mini-batch of data, but I just ran this experiment with 1 agent and only updated once per mini-batch update for simplicity.

### Model


The model consists of 3 main components, a feature extractor, a critic, and an actor. The feature extractor takes in the state of the environment and outputs a feature vector. During training the feature vector is passed to both the critic, which outputs a value prediction of "how good" it is to be in the current state, and the actor, which outputs the next action to take in the environment.

<p align="center">
<img src="https://user-images.githubusercontent.com/42706447/173233189-be2b805a-f96c-42ce-975d-ac424cc515e6.png">
</p>

The main idea behind an actor critic model is that critic parametrizes the value function $V(s)$ which can be used to critique the actor and the actor parametrizes the policy $\pi$ which determines how to act in the environment.

### Feature Extractor

The feature extractor is mostly a convolutional network, with residual connections, and a final linear/flattening layer to form the feature vector.

![fe0](https://user-images.githubusercontent.com/42706447/173233204-512982ae-0e63-469d-b785-8ef10c553189.png)

#### Layers

* Main Block:
  * Conv2d(4, 16, kernel_size=7, stride=2, padding=0)
  * GELU
  * LayerNorm((16, 117, 157))
* l1 Block:
  * Conv2d(16, 32, kernel_size=5, stride=2, padding=0)
* l2 Block:
  * Conv2d(16, 32, kernel_size=5, stride=2, dilation=2, padding=2)
* Main Block (continued):
  * (Output of l1 + l2 block concatenated)
  * GELU
  * ResBlock(64, 128)
  * GELU
  * Conv2d(128, 256, kernel_size=3, stride=2, padding=0)
  * GELU
  * Conv2d(256, 512, kernel_size=3, stride=2, dilation=4, padding=0)
  * GELU
  * Conv2d(512, 512, kernel_size=3, padding=0)
  * GELU
  * AdaptiveAvgPool2d((1, 1))
  * Flatten
  * Linear(512, output_dims)
  * ReLU

### Critic

The critic is a small linear network.

![critic0](https://user-images.githubusercontent.com/42706447/173233217-3f19b2af-907a-4920-baf4-6c40c53ee193.png)

#### Critic Loss

There are two options for the critic loss, a simple MSE loss:

$$
L_{\text{critic}}(V) = (R_t - V_{\theta_k})^2
$$

or a PPO-clipped-like objective:

$$
L_{\text{critic}}(V) = (R_t - V_{\text{clip}} )^2
$$

Where $V_{\text{clip}}$ is defined as:

$$
V_{\text{clip}} = V_{\theta_{k-1}} + \text{clip}(V_{\theta_k} - V_{\theta_{k-1}}, -\epsilon_c, +\epsilon_c)
$$

In practice, we actually take the maximum between the two giving us:

$$
L_{\text{critic}}(V) = max\left[(R_t - V_{\theta_k})^2, (R_t - V_{\text{clip}} )^2\right]
$$

#### Layers

* Main Network:
  * Linear(512, 256)
  * ReLU
  * Linear(256, 1)

### Actor

The actor consists of two modules, one that corresponds to a categorical actor which predicts which button to press (essentially which key to press on the keyboard and/or whether or not to click the mouse button), and a gaussian actor which predicts how to move the mouse. Both modules are small linear networks.

![actor0](https://user-images.githubusercontent.com/42706447/173233222-dea73ba5-5743-408e-9d9f-e320cb6fec13.png)

#### Actor Loss

The actor loss corresponds to the PPO-clip loss with the following objective function:

$$
L_{\text{actor}}(\theta) = E_t = \min \left( \\
\begin{array}{l}
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s, a), g(\epsilon_a, A^{\pi_{\theta_k}}(s, a))
\end{array}
\right)
$$

where $g$ is defined as:

$$
g(\epsilon_{a} , A) =
\begin{cases}
  (1 + \epsilon_{a} ) A &  \text{if } A \ge 0 \\
  (1 - \epsilon_{a} ) A &  \text{if } A < 0
\end{cases}
$$

Where $A$ is defined as the advantage function, which we estimate using [High-dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf)

Since we have 2 sub-actors the loss is taken for both (and later averaged) giving us:

$$
L_{\text{actor}} = L_{\text{gactor}} + L_{\text{cactor}}
$$

as the entire objective function for the actor. ($L_{\text{gactor}}$ is for the gaussian actors contribution and $L_{\text{cactor}}$ is for the categorical actors contribution)

#### Layers

* Gaussian Actor:
  * Linear(512, 256)
  * Tanh
  * Linear(256, 2)

* Categorical Actor:
  * Linear(512, 256)
  * Tanh
  * Linear(256, 1)

### Overall Objective

The complete objective function is defined as:

$$
L(\theta) = \eta_a L_{\text{actor}} + \eta_c L_{\text{critic}} + \eta_s S
$$

Where $S$ is an entropy term defined below and $\eta_{\{x\}}$ is a coefficient scaling the importance of each of the terms. We used $\eta_a = 0.5$ to average the 2 sub-actors loss together, $\eta_c = 1$, and $\eta_s = 0.00001$.

The entropy term $S$ is a bonus term added to encourage exploration, and is defined as the entropy of each of the sub actors distributions averaged together:

$$
S = \frac{S_{\text{gactor}} + S_{\text{cactor}}}{2}
$$

Where $S_{\text{gactor}}$ is the entropy for the gaussian distribution: $-(\frac{1}{2} + \frac{1}{2} \log (2\pi \sigma^2))$ and $S_{\text{cactor}}$ is the entropy for the categorical distribution: $- \log P(X)$

### Observation/State Space

The state space, like typical RL environments, consist of multiple observation frames "stacked" together to give the agent a temporal perspective of the environment. The frames are greyscaled which gives the shape of $[H, W]$ and finally 4 of them are stacked together to make a shape of $[4, H, W]$. This is the only input to our model. Here is an actual example of exactly what the agent is given to work with.

<figure><div>
<img src="https://user-images.githubusercontent.com/42706447/173233261-93c4270b-485d-4a82-b024-fec0a3b5f204.png" class="center" style="width: 49% !important">
<img src="https://user-images.githubusercontent.com/42706447/173233263-7e33ea3a-99ca-49cd-a305-7ad3c251ae91.png" class="center" style="width: 49% !important">
 </div>
 </figure>
<figure><div>
<img src="https://user-images.githubusercontent.com/42706447/173233264-a16f73a8-d217-42e8-889a-8f3f6059f9b5.png" class="center" style="width: 49% !important">
<img src="https://user-images.githubusercontent.com/42706447/173233266-43bd34ff-9500-46e1-ac7b-95b964ad921f.png" class="center" style="width: 49% !important">
 </div>
</figure>

### Action Space

As mentioned the action space reflects the same devices humans use to play the game, consisting of a keyboard, and a mouse. This results in 2 sub-actors for each device, that parameterize each device with their own distribution. The keyboard corresponds to a categorical distribution since it is a binary decision (either the button is pressed or its not) and the mouse corresponds to a gaussian distribution since it consist of a continuous range of possibilities of what angle and magnitude to move the mouse.

#### Categorical Actor

The categorical actor outputs which button to press. There are 6 buttons to choose from that correspond to a particular encoding which maps to in-game actions:

| Buttons | Encoded | Description |
| --- | --- | --- |
| Button 0 | [0, 0, 0, 0, 0, 0] | No attack/movement |
| Button 1 | [0, 1, 0, 0, 0, 0] | Attack $\sim$ Mouse click |
| Button 2 | [0, 0, 1, 0, 0, 0] | Move forward $\sim$ Keypress "W"  |
| Button 3 | [0, 0, 0, 1, 0, 0] | Move left $\sim$ Keypress "A"  |
| Button 4 | [0, 0, 0, 0, 1, 0] | Move backward $\sim$ Keypress "S"  |
| Button 5 | [0, 0, 0, 0, 0, 1] | Move right $\sim$ Keypress "D" |

The encodings map to these in-game actions:
* `TURN_LEFT_RIGHT_DELTA`
* `ATTACK`
* `MOVE_FORWARD`
* `MOVE_LEFT`
* `MOVE_BACKWARD`
* `MOVE_RIGHT`

#### Gaussian Actor

The gaussian actor outputs a mean and standard deviation which creates a gaussian distribution which is sampled from to act as the angle (in degrees) of which the viewing angle will change. This effectively translates into mouse movement from left to right, and we put this value at index 0 of the encoded action that the categorical actor chose and send it to the emulator. An example action, where the agent physically moves to the right and changes the viewing angle to the right by 10 degrees would consist of [10, 0, 0, 0, 0, 1]

#### Multiple Buttons

Note that the agent can't move + shoot + move the mouse at the same time, in one single step, due to the categorical actor only selecting one button. I did train one variation of the agent which was allowed to to this by adding 4 additional buttons in which combined the shooting + movement action (EX: [x, 1, 0, 0, 1, 0]) but to simplify this explanation we'll work with the simpler variant.

I also hypothesized one could make a variation of the algorithm where each button corresponds to a single categorical actor in which it itself is given a binary decision of whether or not to press the button, but have not implemented or done anything with this yet.

## Results

After tweaking hyperparameters a little bit you can produce an agent that achieves, on average, a perfect score for the Defend the Center scenario. This score being 25 due to the agent only being given 26 ammo and thus can only kill 26 monsters, earning 26 points, before dying which takes away 1 point.

### Losses

<div>
   <img src="https://user-images.githubusercontent.com/42706447/173233289-e6108f70-0ce3-4031-9718-b81624732ae5.png" style="width: 49% !important">
   <img src="https://user-images.githubusercontent.com/42706447/173233290-1b65fb1c-3004-40bf-b21c-f408ae921c47.png" style="width: 49% !important">
</div>
<div>
   <img src="https://user-images.githubusercontent.com/42706447/173233291-d7ec2565-e464-4e84-b3cc-55935d1e9a5e.png" style="width: 49% !important">
   <img src="https://user-images.githubusercontent.com/42706447/173233292-4af73507-b82d-4ed1-8514-089220102346.png" style="width: 49% !important">
</div>

### Training

<div>
   <img src="https://user-images.githubusercontent.com/42706447/173233321-b0affbc9-884a-49a0-8510-64b1e44e4724.png" style="width: 49% !important">
   <img src="https://user-images.githubusercontent.com/42706447/173233323-c2899923-7439-4645-8767-80729e7de9d7.png" style="width: 49% !important">
</div>
<div>
   <img src="https://user-images.githubusercontent.com/42706447/173233324-1d920ffb-ca2c-42ab-becb-8cb91a4eb78f.png" style="width: 49% !important">
   <img src="https://user-images.githubusercontent.com/42706447/173233325-ca704545-664b-4872-8cac-c60a435fb60d.png" style="width: 49% !important">
</div>

### Evaluation

<div>
   <img src="https://user-images.githubusercontent.com/42706447/173233347-e006033b-5ee6-4e9d-8868-555742399142.png" style="width: 49% !important">
   <img src="https://user-images.githubusercontent.com/42706447/173233348-c10b07b3-a06c-40a0-8095-81d4e8ca1ef0.png" style="width: 49% !important">
</div>

Here is the agent playing 3 episodes. It pretty much just becomes an aimbot, and has a particular preference for walking backwards. I show all 4 views of the environment which consist of the screen buffer, a depth buffer, a labeled buffer, and the map. The agent still only sees a sliding window of 4 frames of the screen buffer, which are greyscaled and transformed accordingly, these views are just for fun.

https://user-images.githubusercontent.com/42706447/173233122-ac993c3a-fdeb-4137-ba5a-47b59f7ce14f.mp4

Here is the agents perspective:

https://user-images.githubusercontent.com/42706447/173233160-9d0d0004-482b-4ad8-bda2-11dde53cc0a8.mp4

Since I technically stop the episode once the agent is out of ammo for 4 steps, you will see scores of 26, but this is just because of this timeout feature I have, otherwise the agent continued to run around the map like this at the end of the game:

https://user-images.githubusercontent.com/42706447/173233156-9ca7c885-8c39-4a19-a6b4-fcf92de92366.mp4

Until eventually they died, and for the majority of training this experience isn't useful (thus the timeout feature).

## Hyperparameters

* Number of frames per update: 4096
* Maximum number of frames to train on: 4000000  (Only ~1.5M necessary for optimal policy)
* Maximum episode length: 512
* Initial Learning Rate: 7e-4
* Learning Rate Scheduler: Cosine, annealed to 0 over entire training time (Not necessary)
* GAE Gamma $\gamma$: 0.99
* GAE Lambda $\lambda$: 0.95
* Actor Clip Parameter $\epsilon_a$: 0.2
* Critic/Value Clip Parameter $\epsilon_c$: 1000
* Mini batch size: 256
* k epochs: 1
* Entropy Co-Efficient $\eta_s$: 0.00001
* Value Co-Efficient $\eta_c$: 1
* Actor Co-Efficient $\eta_a$: 0.5
* Down Sample: (240, 320) (i.e. no downsampling)
* Frame Skip: 4
* Frame Stack: 4
* Grad norm clip: 0.5
* Optimizer: AdamW
* Weight Decay: 1e-6
