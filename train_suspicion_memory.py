#!/usr/bin/env python3
"""
Enhanced training script for suspicion learning with memory and live notifications.

This version uses DRQN (Deep Recurrent Q-Network) to maintain memory across episodes,
allowing the agent to potentially learn general suspicion patterns.
"""

import matplotlib
# Try to use interactive backend, fall back to non-interactive if needed
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('Agg')  # Non-interactive fallback
import matplotlib.pyplot as plt
import torch
from itertools import count
import numpy as np
import time
import sys

from suspicion_env import SuspicionEnv
from drqn_agent import DRQNAgent
from pattern_detector import SuspicionPatternDetector


class LiveLogger:
    """Live logging and notification system"""

    def __init__(self):
        self.start_time = time.time()
        self.last_notification = 0
        self.notification_interval = 100  # Episodes between status updates

    def log_episode(self, episode, reward, env_stats, agent_metrics, detector_status):
        """Log episode information with live updates"""
        if episode % 10 == 0:  # Every 10 episodes
            elapsed = time.time() - self.start_time
            print(f"\n[Episode {episode:4d}] Reward: {reward:6.2f} | "
                  f"Time: {elapsed:6.1f}s | Eps: {agent_metrics.get('epsilon', 0):.3f}")

            if env_stats.get('with_bomb', False):
                bomb_status = "PUNISH" if env_stats.get('punish', False) else "ACTIVE"
                print(f"               💣 BOMB {bomb_status}")

        # Major notifications
        if episode - self.last_notification >= self.notification_interval:
            self.notify_major_update(episode, detector_status, agent_metrics)
            self.last_notification = episode

    def notify_major_update(self, episode, detector_status, agent_metrics):
        """Send major status notifications"""
        print(f"\n{'='*60}")
        print(f"🔍 SUSPICION CHECK - Episode {episode}")
        print(f"{'='*60}")
        print(f"Status: {detector_status}")

        if 'recent_avg_reward' in agent_metrics:
            print(f"Recent Average Reward: {agent_metrics['recent_avg_reward']:.2f}")

        if 'high_reward_episodes' in agent_metrics:
            high_count = agent_metrics['high_reward_episodes']
            total = min(100, agent_metrics.get('total_episodes', 0))
            if total > 0:
                high_rate = high_count / total
                print(f"High Reward Rate: {high_rate:.1%} ({high_count}/{total})")

                # Alert if suspicion might be developing
                if high_rate < 0.1 and total >= 50:
                    print("🚨 POSSIBLE SUSPICION DETECTED: Low high-reward rate!")
                elif high_rate > 0.5:
                    print("⚠️  Still taking high rewards frequently")

        print(f"{'='*60}\n")

    def notify_pattern_change(self, change_type, episode, details):
        """Notify about significant pattern changes"""
        print(f"\n🔔 PATTERN ALERT [Episode {episode}]: {change_type}")
        print(f"   Details: {details}\n")


def setup_matplotlib():
    """Setup matplotlib for live plotting"""
    try:
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        plt.tight_layout()
        return fig, axes, True
    except:
        print("Warning: Interactive plotting not available")
        return None, None, False


def update_live_plots(fig, axes, episode_rewards, episode_durations, agent, env_stats_history, episode):
    """Update live plots with current training data"""
    if fig is None:
        return

    # Clear all axes
    for ax in axes.flat:
        ax.clear()

    # Find phase transition points
    bomb_introduction = None
    punishment_start = None
    bomb_removal = None

    for i, stats in enumerate(env_stats_history):
        # Bomb introduction (first time bomb appears)
        if bomb_introduction is None and stats.get('with_bomb', False):
            bomb_introduction = i

        # Punishment start (first time punish appears)
        if punishment_start is None and stats.get('punish', False):
            punishment_start = i

        # Bomb removal (bomb was active but now isn't, after punishment)
        if (bomb_removal is None and punishment_start is not None and
            i > punishment_start + 10 and not stats.get('with_bomb', False)):
            bomb_removal = i

    # Plot 1: Episode rewards
    axes[0, 0].set_title('Episode Rewards & Phase Transitions')
    axes[0, 0].plot(episode_rewards, alpha=0.6, color='blue', linewidth=0.8)
    if len(episode_rewards) >= 50:
        # Moving average
        rewards_smooth = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
        axes[0, 0].plot(range(49, len(episode_rewards)), rewards_smooth, color='red', linewidth=2, label='50-ep average')

    # Add phase transition markers
    if bomb_introduction is not None:
        axes[0, 0].axvline(x=bomb_introduction, color='orange', linestyle='--', linewidth=2,
                          label=f'💣 Bomb Intro ({bomb_introduction})')

    if punishment_start is not None:
        axes[0, 0].axvline(x=punishment_start, color='red', linestyle='--', linewidth=2,
                          label=f'⚡ Punishment ({punishment_start})')

    if bomb_removal is not None:
        axes[0, 0].axvline(x=bomb_removal, color='green', linestyle='--', linewidth=2,
                          label=f'✅ Bomb Removed ({bomb_removal})')

    # Highlight phase backgrounds
    bomb_episodes = [i for i, stats in enumerate(env_stats_history) if stats.get('with_bomb', False)]
    punish_episodes = [i for i, stats in enumerate(env_stats_history) if stats.get('punish', False)]

    if bomb_episodes and not punish_episodes:
        # Bomb active but no punishment yet
        axes[0, 0].axvspan(min(bomb_episodes), len(episode_rewards)-1, alpha=0.15, color='orange')
    elif punish_episodes:
        # Punishment phase
        axes[0, 0].axvspan(min(punish_episodes), len(episode_rewards)-1, alpha=0.15, color='red')

    axes[0, 0].legend(fontsize=8, loc='upper left')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')

    # Plot 2: Episode durations
    axes[0, 1].set_title('Episode Durations')
    axes[0, 1].plot(episode_durations, alpha=0.3, color='green', linewidth=0.5, label='Raw')

    # Add moving average for durations
    if len(episode_durations) >= 30:
        duration_smooth = np.convolve(episode_durations, np.ones(30)/30, mode='valid')
        axes[0, 1].plot(range(29, len(episode_durations)), duration_smooth,
                       color='darkgreen', linewidth=2, label='30-ep average')
        axes[0, 1].legend(fontsize=8)

    # Add phase markers to duration plot too
    if bomb_introduction is not None:
        axes[0, 1].axvline(x=bomb_introduction, color='orange', linestyle='--', alpha=0.5)
    if punishment_start is not None:
        axes[0, 1].axvline(x=punishment_start, color='red', linestyle='--', alpha=0.5)
    if bomb_removal is not None:
        axes[0, 1].axvline(x=bomb_removal, color='green', linestyle='--', alpha=0.5)

    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')

    # Plot 3: Reward distribution (recent episodes)
    recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    axes[1, 0].set_title('Recent Reward Distribution')
    axes[1, 0].hist(recent_rewards, bins=20, alpha=0.7, color='purple')
    axes[1, 0].axvline(x=1.0, color='red', linestyle='--', label='Key reward')
    axes[1, 0].axvline(x=2.0, color='orange', linestyle='--', label='Bomb reward')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')

    # Plot 4: Suspicion metrics
    axes[1, 1].set_title('Suspicion Analysis & Phase Markers')
    agent_metrics = agent.get_suspicion_metrics()

    if not agent_metrics.get('insufficient_data', True):
        # Show high vs low reward episodes over time
        window_size = 50
        high_reward_trend = []
        for i in range(window_size, len(episode_rewards)):
            window = episode_rewards[i-window_size:i]
            high_count = sum(1 for r in window if r > 1.5)
            high_reward_trend.append(high_count / window_size)

        if high_reward_trend:
            axes[1, 1].plot(range(window_size, len(episode_rewards)), high_reward_trend,
                           color='red', label='High Reward Rate', linewidth=2)
            axes[1, 1].axhline(y=0.1, color='green', linestyle=':', alpha=0.7, label='Suspicion Threshold')

            # Add same phase markers as main plot
            if bomb_introduction is not None and bomb_introduction >= window_size:
                axes[1, 1].axvline(x=bomb_introduction, color='orange', linestyle='--', alpha=0.7)
            if punishment_start is not None and punishment_start >= window_size:
                axes[1, 1].axvline(x=punishment_start, color='red', linestyle='--', alpha=0.7)
            if bomb_removal is not None and bomb_removal >= window_size:
                axes[1, 1].axvline(x=bomb_removal, color='green', linestyle='--', alpha=0.7)

            axes[1, 1].legend(fontsize=8)
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('High Reward Rate')
            axes[1, 1].set_ylim(0, 1)
    else:
        axes[1, 1].text(0.5, 0.5, f'Insufficient Data\nfor Suspicion Analysis\n({len(episode_rewards)} episodes)',
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=10)

    plt.tight_layout()
    plt.pause(0.01)


def train_suspicion_agent_memory(num_episodes=2000, config=None, verbose=True):
    """
    Train DRQN agent with memory for suspicion learning.
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig, axes, has_interactive = setup_matplotlib()

    # Initialize environment and agent
    env = SuspicionEnv()
    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    # DRQN-specific config - optimized for faster learning
    drqn_config = {
        'batch_size': 32,
        'gamma': 0.99,
        'eps_start': 0.9,
        'eps_end': 0.05,
        'eps_decay': 800,  # Faster decay - less exploration, more exploitation
        'tau': 0.005,
        'lr': 8e-4,  # Slightly higher learning rate
        'memory_capacity': 5000,
        'hidden_size': 32,  # Smaller LSTM for faster training
        'num_layers': 1
    }

    if config:
        drqn_config.update(config)

    agent = DRQNAgent(n_observations, n_actions, device=device, config=drqn_config)
    pattern_detector = SuspicionPatternDetector()
    logger = LiveLogger()

    # Training tracking
    episode_rewards = []
    episode_durations = []
    env_stats_history = []

    if verbose:
        print(f"🚀 Starting MEMORY-BASED suspicion training")
        print(f"Episodes: {num_episodes} | Device: {device}")
        print(f"Environment: {n_observations} observations, {n_actions} actions")
        print(f"DRQN Config: Hidden size {drqn_config['hidden_size']}, Memory {drqn_config['memory_capacity']}")
        print("="*60)

    for i_episode in range(num_episodes):
        # Reset episode
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        agent.reset_episode()  # Don't reset hidden state - keep memory!

        total_reward = 0
        loss_sum = 0
        loss_count = 0

        for t in count():
            # Select and perform action
            action = agent.select_action(state, env)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward_tensor = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store transition
            agent.store_transition(state, action, next_state, reward_tensor)

            # Move to next state
            state = next_state

            # Optimize model
            loss = agent.optimize_model()
            if loss is not None:
                loss_sum += loss
                loss_count += 1

            # Update target network
            agent.update_target_network()

            if done:
                episode_durations.append(t + 1)
                episode_rewards.append(total_reward)
                agent.record_episode_reward(total_reward)

                # Record episode for pattern detection
                env_stats = env.get_episode_stats()
                env_stats_history.append(env_stats)
                pattern_detector.record_episode(
                    env_stats,
                    env_stats['actions_taken'],
                    total_reward,
                    t + 1
                )

                # Live logging
                agent_metrics = agent.get_suspicion_metrics()
                agent_metrics['epsilon'] = agent.get_current_epsilon()
                detector_status = pattern_detector.get_current_status()

                logger.log_episode(i_episode, total_reward, env_stats, agent_metrics, detector_status)

                # Update plots
                if has_interactive and i_episode % 20 == 0:
                    update_live_plots(fig, axes, episode_rewards, episode_durations,
                                     agent, env_stats_history, i_episode)

                # Check for pattern changes
                if i_episode > 100 and i_episode % 100 == 0:
                    analysis = pattern_detector.analyze_suspicion_learning()
                    if analysis.get('sufficient_data', False):
                        score = analysis.get('suspicion_score', 0)
                        if score > 0.5:
                            logger.notify_pattern_change(
                                "STRONG SUSPICION DETECTED",
                                i_episode,
                                f"Score: {score:.2f}"
                            )

                break

    print('\n🎉 Training Complete!')

    # Final analysis
    final_analysis = pattern_detector.analyze_suspicion_learning()
    agent_metrics = agent.get_suspicion_metrics()

    print("\n" + "="*60)
    print("🔍 FINAL SUSPICION ANALYSIS")
    print("="*60)
    print(f"Final Status: {pattern_detector.get_current_status()}")

    if not agent_metrics.get('insufficient_data', True):
        print(f"Recent Average Reward: {agent_metrics['recent_avg_reward']:.2f}")
        print(f"High Reward Episodes (recent 100): {agent_metrics['high_reward_episodes']}")
        print(f"Reward Variance: {agent_metrics['reward_variance']:.2f}")

    # Keep plot open
    if has_interactive:
        update_live_plots(fig, axes, episode_rewards, episode_durations,
                         agent, env_stats_history, num_episodes)
        plt.ioff()
        plt.show()

    return agent, env, pattern_detector, episode_rewards, episode_durations


def main():
    """Main training function"""
    config = {
        'hidden_size': 32,  # Even more laptop-friendly size
        'batch_size': 32,
        'lr': 8e-4,  # Higher learning rate for faster convergence
        'eps_decay': 800  # Less exploration in this sparse environment
    }

    num_episodes = 4000  # Need more episodes for full cycle with new timing

    try:
        agent, env, detector, rewards, durations = train_suspicion_agent_memory(
            num_episodes=num_episodes,
            config=config,
            verbose=True
        )

        # Save model
        agent.save_model('suspicion_drqn_agent.pth')
        print(f"\n💾 Model saved to 'suspicion_drqn_agent.pth'")

        return agent, env, detector

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return None, None, None


if __name__ == "__main__":
    main()