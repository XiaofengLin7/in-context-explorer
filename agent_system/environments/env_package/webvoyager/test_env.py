# Test if env works

test_gym_env = True

if test_gym_env:
    from agent_system.environments.env_package.webvoyager.webgym import WebVoyagerEnv
    import json
    from omegaconf import OmegaConf
    
    env = WebVoyagerEnv(
            api_key="your-api-key-here",
            headless=True,
            text_only=False,
            env_config=OmegaConf.load("agent_system/environments/env_package/webvoyager/configs/webarena_configs.yaml"),
            save_accessibility_tree=True,
            batch_id=0,
            num_containers_per_machine=1,
            worker_id=None
        )

    # Example task
    #task = {"web_name": "Cambridge Dictionary", "id": "Cambridge Dictionary--29", "ques": "Go to the Plus section of Cambridge Dictionary, find Image quizzes and do an easy quiz about Animals and tell me your final score.", "web": "https://dictionary.cambridge.org/"}
    data_file = "/projectnb/noc-lab/ylchen/ICX/tasks/webarena_test_data.jsonl"
    tasks = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
    
    task = tasks[2]
    print(task['eval'])

    # Reset environment
    obs, info = env.reset(task)
    print(f"Reset obs: {obs}")

    # Example actions
    actions = [
        {'action_key': 'answer', 'answer_content': 'Sprite'},
        {'action_key': 'scroll', 'element': -1, 'content': 'down'},
        {'action_key': 'wait'},
        {'action_key': 'click', 'element': 1},
        {'action_key': 'type', 'element': 15, 'content': 'test'},
    ]

    # Execute actions
    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: Action {action['action_key']}, Reward {reward}, Done {done}")
        print(f"obs at step {i+1}: {obs}")
        if done:
            print(f"fail obs: {env.fail_obs}")
            break

    # Close environment
    env.close()
else:
    import os
    from agent_system.environments.env_package.webvoyager.envs import WebVoyagerMultiProcessEnv
    from pprint import pprint

    base_dir = os.path.dirname(__file__)
    os.environ["WEBVOYAGER_DATA"] = os.path.expandvars("$HOME/data/webvoyager")
    env = WebVoyagerMultiProcessEnv(
        seed=0,
        env_num=1, # batch size
        group_n=1, # parallel env number
        resources_per_worker={"num_cpus": 1},
        is_train=False,
        config_path=os.path.join(base_dir, 'configs', 'configs.yaml'),
        env_kwargs={
            "headless": True,
            "text_only": True,
        },
    )

    obs_list, info_list = env.reset()
    pprint(obs_list)
    print("reset ok", len(obs_list))
    if info_list:
        print("info keys:", list(info_list[0].keys()))
    env.close()


