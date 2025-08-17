from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.RandomAgent import RandomAgent
from agents.SimpleAgent import SimpleAgent
from agents.SimpleTrackingAgent import SimpleTrackingAgent
from agents.GradientAgent import GradientAgent
import matplotlib.pyplot as plt


def show_results(bandit_results: list, epsilon_values: list) -> None:
    print("\nAverage results")
    print("Step\tReward\tOptimal action (%)")
    average_rewards = []
    optimal_action_percentages =[]
    for result in bandit_results:
        average_rewards.append(result.get_average_rewards())
        optimal_action_percentages.append(result.get_optimal_action_percentage())
    
    legend = epsilon_values
    plt.rcParams.update({
        'font.size': 22,            # base font size
        'axes.titlesize': 26,       # axes title
        'axes.labelsize': 22,       # x and y labels
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
    })

    for average_reward in average_rewards:
        plt.plot(range(NUM_OF_STEPS), average_reward)
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.legend([f"ε = {eps}" for eps in epsilon_values])
    plt.show()

    for optimal_action_percentage in optimal_action_percentages:
        plt.plot(range(NUM_OF_STEPS), optimal_action_percentage)
    plt.xlabel("Step")
    plt.ylabel("Optimal action percentage")
    plt.legend([f"ε = {eps}" for eps in epsilon_values])
    plt.show()

    '''
    for step in range(NUM_OF_STEPS):
        print(f"{step+1}\t{average_rewards[step]:0.3f}\t{optimal_action_percentage[step]:0.3f}")
    '''


if __name__ == "__main__":

    NUM_OF_RUNS = 2000
    NUM_OF_STEPS = 1000
    epsilon_values = [0, 1]
    result_list = []

    for epsilon in epsilon_values:
        results = BanditResults()
        for run_id in range(NUM_OF_RUNS):
            bandit = BanditEnv(seed=run_id, mean=4)
            num_of_arms = bandit.action_space
            agent = GradientAgent(num_of_arms, baseline=epsilon)  # here you might change the agent that you want to use
            best_action = bandit.best_action
            for _ in range(NUM_OF_STEPS):
                action = agent.get_action()
                reward = bandit.step(action)
                agent.learn(action, reward)
                is_best_action = action == best_action
                results.add_result(reward, is_best_action)
            results.save_current_run()
        result_list.append(results)

    show_results(result_list, epsilon_values)
