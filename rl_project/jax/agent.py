import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Dense(64)(x)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)

        critic = nn.Dense(64)(x)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return actor_mean, critic

class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        self.actor_critic = ActorCritic(action_dim)
        params = self.actor_critic.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))

        self.optimizer = optax.adam(learning_rate)
        self.train_state = train_state.TrainState.create(
            apply_fn=self.actor_critic.apply,
            params=params,
            tx=self.optimizer
        )

    def get_action(self, state, key):
        state = jnp.array(state).reshape(1, -1)  # Ensure state is a 2D array
        actor_mean, _ = self.actor_critic.apply(self.train_state.params, state)
        action = actor_mean + jax.random.normal(key, actor_mean.shape) * 0.1
        return jnp.clip(action, -1, 1).reshape(-1)  # Return a 1D array with a single element

    def update(self, states, actions, rewards, next_states, dones):
        def loss_fn(params):
            actor_mean, critic_value = self.actor_critic.apply(params, states)

            # Compute advantages
            next_critic_value = self.actor_critic.apply(params, next_states)[1]
            td_target = rewards + self.gamma * next_critic_value * (1 - dones)
            td_error = td_target - critic_value

            # Compute actor loss
            log_prob = -0.5 * jnp.sum(jnp.square((actions - actor_mean) / 0.1), axis=-1)
            ratio = jnp.exp(log_prob - old_log_prob)
            actor_loss1 = ratio * td_error
            actor_loss2 = jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * td_error
            actor_loss = -jnp.mean(jnp.minimum(actor_loss1, actor_loss2))

            # Compute critic loss
            critic_loss = jnp.mean(jnp.square(td_error))

            return actor_loss + 0.5 * critic_loss

        old_log_prob = -0.5 * jnp.sum(jnp.square((actions - self.actor_critic.apply(self.train_state.params, states)[0]) / 0.1), axis=-1)

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(self.train_state.params)
        self.train_state = self.train_state.apply_gradients(grads=grads)

        return loss

# Note: This implementation assumes a continuous action space.
# For discrete action spaces, modifications would be needed in the action selection and loss computation.
