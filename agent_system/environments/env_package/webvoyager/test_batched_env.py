


from agent_system.environments.env_package.webvoyager import build_webvoyager_envs, webvoyager_projection
env_kwargs = {}
_envs = build_webvoyager_envs(config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
_val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

projection_f = partial(webvoyager_projection)
envs = WebVoyagerEnvironmentManager(_envs, projection_f, config)
val_envs = WebVoyagerEnvironmentManager(_val_envs, projection_f, config)
return envs, val_envs