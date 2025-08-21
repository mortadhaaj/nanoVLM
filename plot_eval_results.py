#!/usr/bin/env python3
import json
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

def load_eval_results(eval_folder, tasks_to_plot=None):
    """Load all JSON files from the evaluation folder and extract results."""
    json_files = glob.glob(os.path.join(eval_folder, "step_*.json"))
    
    if not json_files:
        print(f"No JSON files found in {eval_folder}")
        return None
    
    results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            step = data.get('global_step', 0)
            metrics = data.get('results', {})
            
            result = {'step': step}
            result.update(metrics)
            
            # Add MME total score if mme is in tasks and both perception and cognition scores exist
            if tasks_to_plot and any('mme_total_score' in task.lower() for task in tasks_to_plot):
                perception_score = result.get('mme_mme_perception_score')
                cognition_score = result.get('mme_mme_cognition_score')
                
                if perception_score is not None and cognition_score is not None:
                    result['mme_total_score'] = perception_score + cognition_score
            
            # Add average score if 'average' is in tasks
            if tasks_to_plot and 'average' in tasks_to_plot:
                # Get only the specified tasks (excluding 'average' and MME-related metrics)
                metrics_to_average = []
                for task in tasks_to_plot:
                    if (task != 'average' and 
                        'mme' not in task.lower() and 
                        task in result and
                        isinstance(result[task], (int, float))):
                        metrics_to_average.append(result[task])
                
                if metrics_to_average:
                    result['average'] = sum(metrics_to_average) / len(metrics_to_average)
            
            results.append(result)
    
    # Sort by step
    results.sort(key=lambda x: x['step'])
    return results

def get_legend_name(eval_folder, custom_name=None):
    """Extract legend name from folder path or use custom name."""
    if custom_name:
        return custom_name
    folder_name = os.path.basename(eval_folder)
    return folder_name.split('_')[-1]

def plot_results(all_results, eval_folders, custom_names=None, tasks_to_plot=None, output_filename=None, steps_to_plot=None):
    """Plot the evaluation results for multiple folders."""
    if not all_results:
        return
    
    # Extract all metric names from all results
    metric_names = set()
    for results in all_results:
        for result in results:
            metric_names.update(k for k in result.keys() if k != 'step')
    
    # Filter metrics based on specified tasks if provided
    if tasks_to_plot:
        filtered_metrics = set()
        for task in tasks_to_plot:
            # Exact match for specified tasks
            if task in metric_names:
                filtered_metrics.add(task)
        metric_names = filtered_metrics
        
        if not metric_names:
            print(f"Warning: No metrics found exactly matching tasks: {tasks_to_plot}")
            return
    
    metric_names = sorted(list(metric_names))
    
    # Create subplots
    n_metrics = len(metric_names)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    _, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Define colors for different runs
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, metric in enumerate(metric_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot each run
        for j, (results, eval_folder) in enumerate(zip(all_results, eval_folders)):
            # Extract values for this metric
            values = []
            metric_steps = []
            missing_steps = []
            
            for result in results:
                # Check if we should include this step
                if steps_to_plot is None or result['step'] in steps_to_plot:
                    if metric in result:
                        values.append(result[metric])
                        metric_steps.append(result['step'])
                    elif steps_to_plot is not None:
                        # Only log missing if specific steps were requested
                        missing_steps.append(result['step'])
            
            # Log missing metrics for specified steps
            if missing_steps:
                folder_name = custom_names[j] if custom_names and custom_names[j] else os.path.basename(eval_folder)
                print(f"Warning: {folder_name} missing '{metric}' for steps: {missing_steps}")
            
            if values:
                custom_name = custom_names[j] if custom_names else None
                legend_name = get_legend_name(eval_folder, custom_name)
                color = colors[j % len(colors)]
                ax.plot(metric_steps, values, marker='o', markersize=2, 
                       color=color, label=legend_name)
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        if len(eval_folders) > 1:
            ax.legend()
    
    # Hide unused subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    # Add title if output filename is specified
    if output_filename:
        plt.suptitle(output_filename, fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Create assets folder if it doesn't exist
    assets_folder = '/fsx/luis_wiedmann/nanoVLM/plots'
    os.makedirs(assets_folder, exist_ok=True)
    
    # Save the plot to assets folder
    if output_filename:
        output_file = os.path.join(assets_folder, f'{output_filename}.png')
    elif len(eval_folders) == 1:
        folder_name = os.path.basename(eval_folders[0])
        output_file = os.path.join(assets_folder, f'{folder_name}_evaluation_plots.png')
    else:
        output_file = os.path.join(assets_folder, 'comparison_evaluation_plots.png')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.close()

def parse_args():
    """Parse command line arguments supporting both folder and folder:name format."""
    parser = argparse.ArgumentParser(
        description='Plot evaluation results from JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python plot_eval_results.py /path/to/eval1
  python plot_eval_results.py Experiment1:/path/to/eval1 Experiment2:/path/to/eval2
  python plot_eval_results.py /path/to/eval1 --tasks vqa gqa
  python plot_eval_results.py Exp1:/path/to/eval1 Exp2:/path/to/eval2 --tasks mmlu"""
    )
    
    parser.add_argument('eval_folders', nargs='+',
                       help='Evaluation folder paths, optionally with custom names (folder:name)')
    parser.add_argument('--tasks', default=['docvqa_val_anls', 'infovqa_val_anls', 'mme_total_score', 'mmmu_val_mmmu_acc', 'mmstar_average', 'ocrbench_ocrbench_accuracy', 'scienceqa_exact_match', 'textvqa_val_exact_match', 'ai2d_exact_match', 'chartqa_relaxed_overall', 'average'], nargs='+',
                       help='Specific tasks to plot (filters metrics containing these task names)')
    parser.add_argument('--output', type=str,
                       help='Custom filename for the saved plot (without extension)')
    parser.add_argument('--steps', nargs='+', type=int,
                       help='Specific steps to plot (e.g., --steps 1000 2000 5000). If not specified, plots all available steps.')
    
    args = parser.parse_args()
    
    eval_folders = []
    custom_names = []
    
    for arg in args.eval_folders:
        if ':' in arg:
            name, folder = arg.rsplit(':', 1)
            eval_folders.append(folder)
            custom_names.append(name)
        else:
            eval_folders.append(arg)
            custom_names.append(None)
    
    # Check if all folders exist
    for eval_folder in eval_folders:
        if not os.path.exists(eval_folder):
            print(f"Error: Folder {eval_folder} does not exist")
            sys.exit(1)
    
    return eval_folders, custom_names, args.tasks, args.output, args.steps

def main():
    eval_folders, custom_names, tasks_to_plot, output_filename, steps_to_plot = parse_args()
    
    # Load results from all folders
    all_results = []
    for eval_folder in eval_folders:
        print(f"Loading evaluation results from: {eval_folder}")
        results = load_eval_results(eval_folder, tasks_to_plot)
        if results:
            print(f"Found {len(results)} evaluation steps")
            
            # Check for missing evaluation steps if specific steps are requested
            if steps_to_plot:
                available_steps = {result['step'] for result in results}
                missing_steps = [step for step in steps_to_plot if step not in available_steps]
                
                if missing_steps:
                    folder_name = custom_names[eval_folders.index(eval_folder)] if custom_names and custom_names[eval_folders.index(eval_folder)] else os.path.basename(eval_folder)
                    print(f"Warning: {folder_name} missing evaluation steps: {missing_steps}")
            
            # Check for missing evaluations if specific tasks are requested
            if tasks_to_plot:
                available_metrics = set()
                for result in results:
                    available_metrics.update(k for k in result.keys() if k != 'step')
                
                missing_tasks = []
                for task in tasks_to_plot:
                    if task not in available_metrics:
                        missing_tasks.append(task)
                
                if missing_tasks:
                    folder_name = custom_names[eval_folders.index(eval_folder)] if custom_names and custom_names[eval_folders.index(eval_folder)] else os.path.basename(eval_folder)
                    print(f"Warning: {folder_name} does not have evaluation for tasks: {missing_tasks}")
            
            all_results.append(results)
        else:
            print(f"No evaluation results found in {eval_folder}")
            all_results.append([])
    
    if any(all_results):
        plot_results(all_results, eval_folders, custom_names, tasks_to_plot, output_filename, steps_to_plot)
    else:
        print("No evaluation results found in any folder")

if __name__ == "__main__":
    main()