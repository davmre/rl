import jax

from daves_rl_lib.algorithms import agent_lib
from daves_rl_lib.environments import environment_lib


def jax_driver(env: environment_lib.Environment, agent: agent_lib.Agent):

    def advance_env(state, weights, seed):
        state = env.reset_if_done(state)
        action = agent.action_dist(weights, state.observation).sample(seed=seed)
        next_state = env.step(state, action)
        return state, action, next_state

    def step(state, weights, seed):
        seed, next_seed = jax.random.split(seed, 2)

        batch_shape = state.done.shape
        if batch_shape:
            advance_envs = jax.vmap(advance_env,
                                    in_axes=(0, None, 0))  # type: ignore
            state, action, next_state = advance_envs(
                state, weights, jax.random.split(seed, batch_shape[0]))
        else:
            state, action, next_state = advance_env(state, weights, seed)

        next_weights = agent.update(weights,
                                    transition=environment_lib.Transition(
                                        observation=state.observation,
                                        action=action,
                                        next_observation=next_state.observation,
                                        reward=next_state.reward,
                                        done=next_state.done))
        return next_state, next_weights, next_seed

    return step


def stateful_driver(env: environment_lib.ExternalEnvironment,
                    agent: agent_lib.Agent,
                    jit_compile=True):

    select_action = lambda w, obs, s: agent.action_dist(w, obs).sample(seed=s)
    update_weights = agent.update

    if jit_compile:
        select_action = jax.jit(select_action)
        update_weights = jax.jit(update_weights)

    def step(state, weights, seed):
        seed, next_seed = jax.random.split(seed, 2)
        if state.done:
            state = env.reset()
        action = select_action(weights, state.observation, seed)
        next_state = env.step(action)
        transition = environment_lib.Transition(
            state.observation,
            action=action,
            next_observation=next_state.observation,
            reward=next_state.reward,
            done=next_state.done)
        next_weights = update_weights(weights, transition=transition)
        return next_state, next_weights, next_seed

    return step
