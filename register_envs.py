from gymnasium.envs.registration import register

register(
    id="roundabout-yield-exit-v0",                    # unique ID
    entry_point="custom_env:RoundaboutYieldExitEnv",  # module_name:ClassName
)