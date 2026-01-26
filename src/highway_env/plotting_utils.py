import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np


# --- Small guide ---
# The relative performance plot shows the normalized performance of agents in the non-crashed episodes.
# It filters out crashed episodes, computes mean metrics per agent, normalizes them.
#
# In the global comparion for the distance is used the survival time, 
# while in the agent comparison is used the actual distance traveled.

# Configuration Info
ENV_INFO = """Environment:
- Duration: 120s
- Vehicles: 20
- Density: 0.3
- Collision Reward: -1.5
- Speed Reward Range: [0, 35]"""

MODEL_INFO = """Model (DQN):
- Hidden Layers: [256, 256]
- Buffer Size: 50k
- Learning Rate: 1e-4
- Gamma: 0.95"""

RESULTS_DIR = os.path.join("results", "highway_dqn_results")

def find_latest_file(mode, total_steps=None):
    """Finds the latest result file for a given mode and optional step count."""
    if total_steps:
        pattern = os.path.join(RESULTS_DIR, f"results_{mode}_{total_steps}_*.csv")
    else:
        # Matches both old format "results_mode_timestamp.csv" and new "results_mode_steps_timestamp.csv"
        pattern = os.path.join(RESULTS_DIR, f"results_{mode}_*.csv")
        
    files = glob.glob(pattern)
    if not files:
        print(f"No results found for mode: {mode} (total_steps={total_steps})")
        return None
        
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def load_data(files_map):
    """
    Loads data from a map of {label: filename_or_path}.
    Returns a list of dicts [{'label': label, 'df': dataframe}].
    """
    data = []
    for label, filename in files_map.items():
        # Check if full path or just filename
        if os.path.dirname(filename):
            filepath = filename
        else:
            filepath = os.path.join(RESULTS_DIR, filename)
            
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        try:
            df = pd.read_csv(filepath)
            # Add label column
            df['Run'] = label
            data.append({
                'label': label,
                'df': df
            })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return data

def plot_metrics(data_list, title="Comparison", plot_type="bar", output_path=None):
    """
    Generic plotter for 2x2 grid of metrics.
    """
    if not data_list:
        print("No data to plot.")
        return

    unique_configs = [d['label'] for d in data_list]
    combined_df = pd.concat([d['df'] for d in data_list], ignore_index=True)
    
    sns.set_theme(style="whitegrid")
    
    if plot_type == "bar":
        fig_width = max(16, 4 * len(unique_configs))
    else:
        fig_width = 16

    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 10))

    metrics = [
        ("reward", "Average Reward", axes[0, 0]),
        ("length", "Average Length (Survival)", axes[0, 1]), # Changed distance to length to match baseline
        ("avg_speed", "Average Speed", axes[1, 0]),
        ("crashed", "Crash Rate", axes[1, 1])
    ]

    for metric, metric_title, ax in metrics:
        if plot_type == "bar":
            if metric == "crashed":
                crash_rates = combined_df.groupby('Run', sort=False)[metric].mean().reindex(unique_configs).reset_index()
                sns.barplot(ax=ax, data=crash_rates, x="Run", y=metric, hue="Run", palette="magma", legend=False)
                ax.set_ylabel("Crash Probability")
            else:
                sns.barplot(ax=ax, data=combined_df, x="Run", y=metric, hue="Run", palette="viridis", legend=False, order=unique_configs)
            ax.set_xlabel("")
            
        elif plot_type == "line":
            if metric == "crashed":
                for i, config in enumerate(unique_configs):
                    subset = combined_df[combined_df['Run'] == config]
                    y_vals = subset[metric].astype(int) + (i * 0.05)
                    ax.scatter(subset['episode'], y_vals, label=config, alpha=0.7)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['False', 'True'])
            else:
                sns.lineplot(ax=ax, data=combined_df, x="episode", y=metric, hue="Run", style="Run", markers=True, dashes=False)
            
            ax.set_xlabel("Episode")
            ax.legend()

        ax.set_title(metric_title)

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        
    plt.show()

def plot_agent_comparison_dots(data_list, title="Agent Comparison (Per Episode)", output_path=None, aggregate_agents=False):
    """
    Plots metrics as lines with markers per episode.
    Uses specific matplotlib loop for Distance as requested.
    """
    if not data_list:
        print("No data to plot.")
        return

    combined_df = pd.concat([d['df'] for d in data_list], ignore_index=True)
    unique_configs = [d['label'] for d in data_list]
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    metrics = [
        ("reward", "Reward per Episode", axes[0, 0]),
        ("distance", "Distance Traveled", axes[0, 1]),
        ("avg_speed", "Avg Speed per Episode", axes[1, 0]),
        ("performance", "Relative Performance (Non-Crashed)", axes[1, 1])
    ]

    for metric, metric_title, ax in metrics:
        if metric == "performance":
            # Filter for Non-Crashed episodes only
            survived_df = combined_df[combined_df['crashed'] == False]
            
            if survived_df.empty:
                ax.text(0.5, 0.5, "No Non-Crashed Episodes", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric_title)
                continue
            
            # Calculate Means grouped by Run AND Agent
            means = survived_df.groupby(['Run', 'agent_id'])[['reward', 'distance', 'avg_speed']].mean()
            
            # Normalize (divide by max of each metric column across all agents/runs)
            normalized = means / means.max()
            
            # Prepare for plotting
            normalized = normalized.reset_index()
            # Create label: just Run name when aggregated, otherwise Run + Agent
            if aggregate_agents:
                normalized['AgentLabel'] = normalized['Run']
            else:
                normalized['AgentLabel'] = normalized['Run'] + " - Ag " + normalized['agent_id'].astype(str)
            
            melted = normalized.melt(id_vars=["AgentLabel"], value_vars=['reward', 'distance', 'avg_speed'], 
                                     var_name="Metric", value_name="Score")
            
            # Plot
            sns.barplot(ax=ax, data=melted, x="Metric", y="Score", hue="AgentLabel", palette="viridis")
            ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Score (Relative to Max)")
            ax.legend(loc='lower right', fontsize='x-small')

        elif metric == "distance":
            # Explicit matplotlib loop for distance
            for config in unique_configs:
                df_run = combined_df[combined_df['Run'] == config]
                
                if aggregate_agents and df_run['agent_id'].nunique() > 1:
                    # Aggregate: average across agents per episode (crashed only if ALL agents crashed)
                    agg_data = df_run.groupby('episode').agg({'distance': 'mean', 'crashed': 'min'}).reset_index()
                    ax.plot(agg_data['episode'], agg_data['distance'], label=config, marker='s', alpha=0.7)
                    crash_data = agg_data[agg_data['crashed'] == True]
                    if not crash_data.empty:
                        ax.scatter(crash_data['episode'], crash_data['distance'], color='red', marker='x', s=100, zorder=5)
                else:
                    # Plot each agent separately
                    agents = sorted(df_run['agent_id'].unique())
                    for agent in agents:
                        agent_data = df_run[df_run['agent_id'] == agent]
                        label = config if len(agents) == 1 else f"{config} - Agent {agent}"
                        ax.plot(agent_data['episode'], agent_data['distance'], label=label, marker='s', alpha=0.7)
                        crash_data = agent_data[agent_data['crashed'] == True]
                        if not crash_data.empty:
                            ax.scatter(crash_data['episode'], crash_data['distance'], color='red', marker='x', s=100, zorder=5)

            ax.set_xlabel("Episode")
            ax.set_ylabel("Distance (m)")
            ax.legend()
            ax.grid(True)
            
        else:
            # Explicit loop for Reward and Speed
            for config in unique_configs:
                df_run = combined_df[combined_df['Run'] == config]
                marker = 'o' if metric == 'reward' else '^'
                
                if aggregate_agents and df_run['agent_id'].nunique() > 1:
                    # Aggregate: average across agents per episode (crashed only if ALL agents crashed)
                    agg_data = df_run.groupby('episode').agg({metric: 'mean', 'crashed': 'min'}).reset_index()
                    ax.plot(agg_data['episode'], agg_data[metric], label=config, marker=marker, alpha=0.7)
                    crash_data = agg_data[agg_data['crashed'] == True]
                    if not crash_data.empty:
                        ax.scatter(crash_data['episode'], crash_data[metric], color='red', marker='x', s=100, zorder=5)
                else:
                    # Plot each agent separately
                    agents = sorted(df_run['agent_id'].unique())
                    for agent in agents:
                        agent_data = df_run[df_run['agent_id'] == agent]
                        label = config if len(agents) == 1 else f"{config} - Agent {agent}"
                        ax.plot(agent_data['episode'], agent_data[metric], label=label, marker=marker, alpha=0.7)
                        crash_data = agent_data[agent_data['crashed'] == True]
                        if not crash_data.empty:
                            ax.scatter(crash_data['episode'], crash_data[metric], color='red', marker='x', s=100, zorder=5)

            ax.set_xlabel("Episode")
            ax.legend()
            ax.grid(True)
            
        ax.set_title(metric_title)

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        
    plt.show()