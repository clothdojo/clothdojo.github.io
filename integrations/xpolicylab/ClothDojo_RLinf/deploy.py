"""Run the historical ten-action chunk schedule through XPolicyLab."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        if len(actions) < 10:
            raise ValueError("RLinf Pi0.5 returned fewer than ten actions")
        for action in actions[:10]:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError("ClothDojo_RLinf supports one simulation world per policy server")
