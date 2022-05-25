import velha
import Proximal_Policy_Optimization


def main():
    # True if you want to render the environment
    render = False

    """
    ## Initializations
    """
    # Initialize the environment and get the dimensionality of the
    # observation space and the number of possible actions
    env = velha.make()
    observation_dimensions = env.observation_space
    num_actions = env.action_space

    # Initialize the observation, episode return and episode length
    observation = env.reset()
    episode_return = 0
    episode_length = 0
    

    # Hyperparameters of the PPO algorithm
    steps_per_epoch = 4000
    epochs = 30
    ppo = Proximal_Policy_Optimization.make(
        observation_dimensions = observation_dimensions,
        hidden_sizes = (64, 64),
        num_actions = num_actions,
        policy_learning_rate = 0.00003,
        value_function_learning_rate = 0.0003,
        steps_per_epoch = 100,
        train_policy_iterations = 80,
        train_value_iterations = 80,
        target_kl = 0.01,
        clip_ratio = 0.2
    )

    """
    ## Run
    """
    # Iterate over the number of epochs
    for epoch in range(epochs):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        # Iterate over the steps of each epoch
        while True:
            if render:
                env.render()
            
            if env.player == 1:
                print("Jogador %s" % env.player)
                x = int(input("Digite a linha: "))
                y = int(input("Digite a coluna: "))

            action = ppo.get_action(observation)
            observation_new, reward, done, info = env.step(action)
            episode_return += reward
            episode_length += 1

            # Store the transition
            ppo.store(reward)

            # Update the observation
            observation = observation_new

            # If the episode is done, reset the environment
            if done:
                ppo.finish_epoch(done, observation)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation = env.reset()
                episode_return = 0
                episode_length = 0
                break
        
        ppo.train()

        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )










if __name__ == "__main__":
    main()
