import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

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

    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 12))
    fig.suptitle(title, fontsize=16, y=0.98)
        
    # Add Info Blocks at the very top
    fig.text(0.15, 0.98, ENV_INFO, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.8'))
    fig.text(0.85, 0.98, MODEL_INFO, fontsize=10, verticalalignment='top', ha='right',
             bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.8'))
    
    # Increase top margin significantly to clear info blocks
    plt.subplots_adjust(top=0.78, hspace=0.4, wspace=0.3)

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

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        
    plt.show()

def plot_agent_comparison_dots(data_list, title="Agent Comparison (Per Episode)", output_path=None):
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Add Info Blocks at the very top
    fig.text(0.15, 0.98, ENV_INFO, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.8'))
    fig.text(0.85, 0.98, MODEL_INFO, fontsize=10, verticalalignment='top', ha='right',
             bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.8'))
    
    # Increase top margin significantly to clear info blocks
    plt.subplots_adjust(top=0.78, hspace=0.4, wspace=0.3)

    metrics = [
        ("reward", "Reward per Episode", axes[0, 0]),
        ("distance", "Distance Traveled", axes[0, 1]),
        ("avg_speed", "Avg Speed per Episode", axes[1, 0]),
        ("performance", "Relative Performance (Non-Crashed)", axes[1, 1])
    ]

    for metric, metric_title, ax in metrics:
        if metric == "performance":
            # 1. Filter for Non-Crashed episodes
            survived_df = combined_df[combined_df['crashed'] == False]
            
            if not survived_df.empty:
                # 2. Calculate Means grouped by Run AND Agent
                means = survived_df.groupby(['Run', 'agent_id'])[['reward', 'distance', 'avg_speed']].mean()
                
                # 3. Normalize (divide by max of each metric column across all agents/runs)
                normalized = means / means.max()
                
                # 4. Prepare for plotting
                normalized = normalized.reset_index()
                # Create a unique label for each agent in each run
                normalized['AgentLabel'] = normalized['Run'] + " - Ag " + normalized['agent_id'].astype(str)
                
                melted = normalized.melt(id_vars=["AgentLabel"], value_vars=['reward', 'distance', 'avg_speed'], 
                                         var_name="Metric", value_name="Score")
                
                # 5. Plot
                sns.barplot(ax=ax, data=melted, x="Metric", y="Score", hue="AgentLabel", palette="viridis")
                ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("Score (Relative to Max)")
                ax.legend(loc='lower right', fontsize='x-small')
            else:
                ax.text(0.5, 0.5, "No Non-Crashed Episodes", ha='center', va='center')

        elif metric == "distance":
            # Explicit matplotlib loop for distance as requested
            for config in unique_configs:
                df_run = combined_df[combined_df['Run'] == config]
                agents = sorted(df_run['agent_id'].unique())
                for agent in agents:
                    agent_data = df_run[df_run['agent_id'] == agent]
                    label = f"{config} - Agent {agent}" if len(unique_configs) > 1 or len(agents) > 1 else f"Agent {agent}"
                    ax.plot(agent_data['episode'], agent_data['distance'], label=label, marker='s', alpha=0.7)
                    
                    # Highlight crashes
                    crash_data = agent_data[agent_data['crashed'] == True]
                    if not crash_data.empty:
                        ax.scatter(crash_data['episode'], crash_data['distance'], color='red', marker='x', s=100, zorder=5)

            ax.set_xlabel("Episode")
            ax.set_ylabel("Distance (m)")
            ax.legend()
            ax.grid(True)
            
        else:
            # Explicit loop for Reward and Speed to avoid CI shadows and ensure clear dot-lines
            for config in unique_configs:
                df_run = combined_df[combined_df['Run'] == config]
                agents = sorted(df_run['agent_id'].unique())
                for agent in agents:
                    agent_data = df_run[df_run['agent_id'] == agent]
                    label = f"{config} - Agent {agent}" if len(unique_configs) > 1 or len(agents) > 1 else f"Agent {agent}"
                    marker = 'o' if metric == 'reward' else '^' # Circle for reward, Triangle for speed
                    ax.plot(agent_data['episode'], agent_data[metric], label=label, marker=marker, alpha=0.7)
                    
                    # Highlight crashes
                    crash_data = agent_data[agent_data['crashed'] == True]
                    if not crash_data.empty:
                        ax.scatter(crash_data['episode'], crash_data[metric], color='red', marker='x', s=100, zorder=5)

            ax.set_xlabel("Episode")
            ax.legend()
            ax.grid(True)
            
        ax.set_title(metric_title)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        
    plt.show()