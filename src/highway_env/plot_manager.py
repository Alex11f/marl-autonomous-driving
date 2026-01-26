import os
import pandas as pd
from plotting_utils import load_data, plot_metrics, plot_agent_comparison_dots

# --- FILE GROUPS (DEFINITIONS) ---
# Define groups of files to be used in plots
FILE_GROUPS = {
    "all_steps": {
        "Single 50k": "results_single_50000_1768404543.csv",
        # "Single 100k": "results_single_1768343914.csv",
        "Single 100k": "results_single_100000_2.csv",
        # "Shared 50k": "results_multi-shared_1768337662.csv",
        "Shared 50k": "results_multi-shared_50000_2.csv",
        # "Shared 100k": "results_multi-shared_100000_1768390911.csv",
        "Shared 100k": "results_multi-shared_100000_3.csv",
        # "Indep 50k": "results_multi-independent_1768341042.csv",
        "Indep 50k": "results_multi-independent_50000_2.csv",
        # "Indep 100k": "results_multi-independent_1768347379.csv",
        "Indep 100k": "results_multi-independent_100000_2.csv",
    },
    "comparison_50k": {
        "Single 50k": "results_single_50000_1768404543.csv",
        "Shared 50k": "results_multi-shared_1768337662.csv",
        "Indep 50k": "results_multi-independent_1768341042.csv",
    },
    "comparison_100k": {
        "Single 100k": "results_single_100000_2.csv",
        "Shared 100k": "results_multi-shared_100000_2.csv",
        "Indep 100k": "results_multi-independent_100000_2.csv",

        # "[256, 256]": "results_multi-shared_100000_2.csv",
        # "[512, 512]": "results_multi-shared_100000_double.csv",
        # "[512, 512, 512]": "results_multi-shared_100000_triple.csv",

        
    },
    "shared_100k": {
        "Shared 100k": "results_multi-shared_100000_3.csv",
    },
    "indep_100k": {
        # "Indep 100k": "results_multi-independent_1768347379.csv",
        "Indep 100k": "results_multi-independent_100000_2.csv",
    },
    "shared_vs_indep_100k": {
        "Shared 100k": "results_multi-shared_100000_3.csv",
        "Indep 100k": "results_multi-independent_100000_2.csv",
    }
}

# --- PLOT CONFIGURATION ---
# Add or comment out entries here to generate specific plots
PLOTS_TO_GENERATE = [
    # 1. Whole Comparison (Bar)
    {
        "name": "Global Comparison (Bar)",
        "files": FILE_GROUPS["all_steps"],
        "type": "bar",
        "agent": None  # All agents aggregated
    },
    # 2. Whole Comparison (Dot)
    {
        "name": "Comparison at 50k Steps",
        "files": FILE_GROUPS["comparison_50k"],
        "type": "dot",
        "agent": [0, 1] # Compare Agent 0 and 1 across all strategies
    },
    # 2b. Whole Comparison at 100k Steps (Dot)
    {
        "name": "Global Comparison at 100k Steps",
        "files": FILE_GROUPS["comparison_100k"],
        "type": "dot",
        "agent": [0, 1], # Compare Agent 1 across all strategies
        "aggregate_agents": True
    },
    # 3. Agent Comparison: Shared (0 vs 1)
    {
        "name": "Shared Agents Comparison (0 vs 1)",
        "files": FILE_GROUPS["shared_100k"],
        "type": "dot",
        "agent": None # Show all agents (0 and 1)
    },
    # 4. Agent Comparison: Independent (0 vs 1)
    {
        "name": "Independent Agents Comparison (0 vs 1)",
        "files": FILE_GROUPS["indep_100k"],
        "type": "dot",
        "agent": None # Show all agents
    },
    # 5. Mode Comparison: Shared 0 vs Independent 0
    {
        "name": "Shared vs Independent (Agent 0)",
        "files": FILE_GROUPS["shared_vs_indep_100k"],
        "type": "dot",
        "agent": [0, 1], # Filter only Agent 0
        "aggregate_agents": True
    },
    # 6. Mean Performance (Ag0 & Ag1)
    {
        "name": "Global Mean Performance (Ag0 & Ag1)",
        "files": FILE_GROUPS["comparison_100k"],
        "type": "dot",
        "agent": [0, 1], # Aggregate mean of Agent 0 and 1
        "aggregate_agents": True
    },
]

def main():
    print(f"Generating {len(PLOTS_TO_GENERATE)} plots...")

    for config in PLOTS_TO_GENERATE:
        print(f"\nProcessing: {config['name']}")
        
        # Load data
        data_list = load_data(config['files'])
        if not data_list:
            print("  -> No data found, skipping.")
            continue

        # Filter by Agent
        target_agent = config.get('agent')
        is_list_of_agents = isinstance(target_agent, list)
        
        if target_agent is not None:
            print(f"  -> Filtering for Agent(s) {target_agent}")
            filtered_list = []
            for item in data_list:
                df = item['df']
                if 'agent_id' in df.columns:
                    if is_list_of_agents:
                        # Keep rows where agent_id is in the list
                        filtered_df = df[df['agent_id'].isin(target_agent)].copy()
                    else:
                        # Keep rows where agent_id equals the value
                        filtered_df = df[df['agent_id'] == target_agent].copy()
                        
                    if not filtered_df.empty:
                        filtered_list.append({'label': item['label'], 'df': filtered_df})
                else:
                    # If no agent_id (e.g. single agent), keep it
                    filtered_list.append(item)
            data_list = filtered_list

        if not data_list:
            print("  -> No data remaining after filtering.")
            continue

        # Generate Plot
        plot_type = config.get('type', 'bar')
        output_name = config['name'].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("&", "and") + ".png"
        output_path = os.path.join("results", "highway_dqn_results", output_name)
        
        # Ensure output directory exists (it should, but just in case)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if plot_type == "dot":
            # Use aggregate_agents from config if specified, otherwise default based on agent list
            aggregate = config.get('aggregate_agents', is_list_of_agents)
            plot_agent_comparison_dots(data_list, title=config['name'], output_path=None, aggregate_agents=aggregate) 
        else:
            plot_metrics(data_list, title=config['name'], plot_type=plot_type, output_path=None)

if __name__ == "__main__":
    main()
