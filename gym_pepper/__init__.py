from gym.envs.registration import register

register(
    id='pepper-v0',
    entry_point='gym_pepper.envs:PepperEnv',
)
register(
    id='pepper-extrahard-v0',
    entry_point='gym_foo.envs:PepperExtraHardEnv',
)
