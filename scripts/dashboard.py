#!/usr/bin/env python3
"""
Dashboard for monitoring agent performance in Cities Skylines 2.

This script creates a Streamlit dashboard that displays real-time metrics
and visualizations of the agent's performance.
"""

import os
import sys
import json
import glob
import time
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dashboard for monitoring agent performance")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing training logs")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the dashboard on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to run the dashboard on")
    parser.add_argument("--refresh_interval", type=int, default=30, help="Dashboard refresh interval in seconds")
    return parser.parse_args()


def load_training_data(log_dir):
    """Load training data from logs.
    
    Args:
        log_dir: Directory containing training logs
        
    Returns:
        Dictionary of training data
    """
    # Find all training log files
    log_files = glob.glob(os.path.join(log_dir, "**", "training_log.json"), recursive=True)
    
    if not log_files:
        return None
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Load the most recent log file
    with open(log_files[0], "r") as f:
        training_data = json.load(f)
    
    return training_data


def load_checkpoint_data(log_dir):
    """Load checkpoint data.
    
    Args:
        log_dir: Directory containing checkpoints
        
    Returns:
        List of checkpoint info
    """
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(log_dir, "**", "checkpoint_*.pt"), recursive=True)
    
    if not checkpoint_files:
        return []
    
    # Get info about each checkpoint
    checkpoint_info = []
    
    for cp_file in checkpoint_files:
        # Extract checkpoint number
        cp_filename = os.path.basename(cp_file)
        try:
            cp_num = int(cp_filename.split("_")[1].split(".")[0])
        except:
            cp_num = 0
        
        # Get file stats
        stats = os.stat(cp_file)
        
        checkpoint_info.append({
            "file": cp_file,
            "number": cp_num,
            "size_mb": stats.st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(stats.st_mtime),
            "created": datetime.fromtimestamp(stats.st_ctime),
        })
    
    # Sort by checkpoint number
    checkpoint_info.sort(key=lambda x: x["number"])
    
    return checkpoint_info


def load_eval_data(log_dir):
    """Load evaluation data from logs.
    
    Args:
        log_dir: Directory containing evaluation logs
        
    Returns:
        Dictionary of evaluation data
    """
    # Find all evaluation log files
    eval_files = glob.glob(os.path.join(log_dir, "**", "eval_*.json"), recursive=True)
    
    if not eval_files:
        return None
    
    # Sort by name (which includes timestamp)
    eval_files.sort()
    
    # Load all evaluation data
    eval_data = []
    
    for eval_file in eval_files:
        with open(eval_file, "r") as f:
            data = json.load(f)
            # Extract timestamp from filename
            filename = os.path.basename(eval_file)
            try:
                timestamp = filename.split("_")[1].split(".")[0]
                data["timestamp"] = timestamp
            except:
                data["timestamp"] = "unknown"
            
            eval_data.append(data)
    
    return eval_data


def create_episode_metrics_chart(training_data):
    """Create chart for episode metrics.
    
    Args:
        training_data: Dictionary of training data
        
    Returns:
        Plotly figure
    """
    # Create DataFrame
    episodes = list(range(1, len(training_data["episode_rewards"]) + 1))
    df = pd.DataFrame({
        "Episode": episodes,
        "Reward": training_data["episode_rewards"],
        "Length": training_data["episode_lengths"],
    })
    
    # Create subplot with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=df["Episode"], y=df["Reward"], name="Reward", mode="lines+markers"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df["Episode"], y=df["Length"], name="Length", mode="lines+markers"),
        secondary_y=True,
    )
    
    # Set titles
    fig.update_layout(title_text="Episode Metrics")
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Reward", secondary_y=False)
    fig.update_yaxes(title_text="Length", secondary_y=True)
    
    return fig


def create_loss_chart(training_data):
    """Create chart for loss metrics.
    
    Args:
        training_data: Dictionary of training data
        
    Returns:
        Plotly figure
    """
    # Check if loss data is available
    if "actor_losses" not in training_data or not training_data["actor_losses"]:
        return None
    
    # Create DataFrame
    updates = list(range(1, len(training_data["actor_losses"]) + 1))
    df = pd.DataFrame({
        "Update": updates,
        "Actor Loss": training_data["actor_losses"],
        "Critic Loss": training_data["critic_losses"],
        "Entropy": training_data["entropies"],
    })
    
    # Create subplot with three y-axes
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                      subplot_titles=("Actor Loss", "Critic Loss", "Entropy"))
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=df["Update"], y=df["Actor Loss"], name="Actor Loss", mode="lines"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df["Update"], y=df["Critic Loss"], name="Critic Loss", mode="lines"),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df["Update"], y=df["Entropy"], name="Entropy", mode="lines"),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(height=600, title_text="Training Losses", showlegend=False)
    fig.update_xaxes(title_text="Update", row=3, col=1)
    
    return fig


def create_action_distribution_chart(training_data):
    """Create chart for action distribution.
    
    Args:
        training_data: Dictionary of training data
        
    Returns:
        Plotly figure
    """
    # Check if action counts are available
    if "action_counts" not in training_data or not training_data["action_counts"]:
        return None
    
    # Convert action counts to DataFrame
    action_counts = training_data["action_counts"]
    
    # Convert dictionary to lists
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    
    # Create DataFrame
    df = pd.DataFrame({
        "Action": actions,
        "Count": counts,
    })
    
    # Create pie chart
    fig = px.pie(df, values="Count", names="Action", title="Action Distribution")
    
    return fig


def create_eval_metrics_chart(eval_data):
    """Create chart for evaluation metrics.
    
    Args:
        eval_data: List of evaluation data
        
    Returns:
        Plotly figure
    """
    if not eval_data:
        return None
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            "Timestamp": data["timestamp"],
            "Mean Reward": data["mean_reward"],
            "Min Reward": data["min_reward"],
            "Max Reward": data["max_reward"],
            "Mean Length": data["mean_length"],
        }
        for data in eval_data
    ])
    
    # Create subplot with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=df["Timestamp"], y=df["Mean Reward"], name="Mean Reward", mode="lines+markers"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df["Timestamp"], y=df["Min Reward"], name="Min Reward", mode="lines+markers", 
                 line=dict(dash="dash")),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df["Timestamp"], y=df["Max Reward"], name="Max Reward", mode="lines+markers",
                 line=dict(dash="dash")),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df["Timestamp"], y=df["Mean Length"], name="Mean Length", mode="lines+markers"),
        secondary_y=True,
    )
    
    # Set titles
    fig.update_layout(title_text="Evaluation Metrics Over Time")
    fig.update_xaxes(title_text="Evaluation")
    fig.update_yaxes(title_text="Reward", secondary_y=False)
    fig.update_yaxes(title_text="Length", secondary_y=True)
    
    return fig


def create_dashboard(log_dir, refresh_interval=30):
    """Create Streamlit dashboard.
    
    Args:
        log_dir: Directory containing logs
        refresh_interval: Dashboard refresh interval in seconds
    """
    # Set page title
    st.set_page_config(
        page_title="Cities Skylines 2 Agent Dashboard",
        page_icon="🏙️",
        layout="wide",
    )
    
    # Create sidebar
    st.sidebar.title("Cities Skylines 2")
    st.sidebar.subheader("Agent Performance Dashboard")
    
    # Log directory input
    log_dir_input = st.sidebar.text_input("Log Directory", value=log_dir)
    
    # Auto-refresh checkbox
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    
    # Refresh button
    if st.sidebar.button("Refresh Now"):
        st.experimental_rerun()
    
    # Auto-refresh
    if auto_refresh:
        refresh_interval_input = st.sidebar.slider("Refresh Interval (seconds)", 5, 300, refresh_interval)
        st.sidebar.write(f"Next refresh in {refresh_interval_input} seconds")
        time.sleep(refresh_interval_input)
        st.experimental_rerun()
    
    # Main content
    st.title("Cities Skylines 2 Agent Dashboard")
    
    # Load data
    training_data = load_training_data(log_dir_input)
    checkpoint_data = load_checkpoint_data(log_dir_input)
    eval_data = load_eval_data(log_dir_input)
    
    # Show last update time
    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Training status
    if training_data:
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Display metrics
        col1.metric("Episodes", len(training_data["episode_rewards"]))
        col2.metric("Latest Reward", f"{training_data['episode_rewards'][-1]:.2f}")
        col3.metric("Avg Reward (Last 10)", f"{np.mean(training_data['episode_rewards'][-10:]):.2f}")
        col4.metric("Latest Episode Length", f"{training_data['episode_lengths'][-1]}")
        
        # Create tabs for different charts
        tab1, tab2, tab3, tab4 = st.tabs(["Episode Metrics", "Training Losses", "Action Distribution", "Evaluation"])
        
        with tab1:
            # Episode metrics chart
            episode_fig = create_episode_metrics_chart(training_data)
            st.plotly_chart(episode_fig, use_container_width=True)
        
        with tab2:
            # Loss chart
            loss_fig = create_loss_chart(training_data)
            if loss_fig:
                st.plotly_chart(loss_fig, use_container_width=True)
            else:
                st.info("No loss data available yet")
        
        with tab3:
            # Action distribution chart
            action_fig = create_action_distribution_chart(training_data)
            if action_fig:
                st.plotly_chart(action_fig, use_container_width=True)
            else:
                st.info("No action distribution data available yet")
        
        with tab4:
            # Evaluation metrics chart
            if eval_data:
                eval_fig = create_eval_metrics_chart(eval_data)
                st.plotly_chart(eval_fig, use_container_width=True)
            else:
                st.info("No evaluation data available yet")
        
        # Checkpoints section
        st.subheader("Checkpoints")
        
        if checkpoint_data:
            # Convert to DataFrame for display
            checkpoints_df = pd.DataFrame(checkpoint_data)
            # Format columns
            checkpoints_df["modified"] = checkpoints_df["modified"].dt.strftime("%Y-%m-%d %H:%M:%S")
            checkpoints_df["created"] = checkpoints_df["created"].dt.strftime("%Y-%m-%d %H:%M:%S")
            checkpoints_df["size_mb"] = checkpoints_df["size_mb"].round(2)
            # Display as table
            st.dataframe(checkpoints_df[["number", "file", "size_mb", "modified"]], hide_index=True)
        else:
            st.info("No checkpoints available yet")
    
    else:
        st.warning(f"No training data found in {log_dir_input}")
        st.info("Please make sure the log directory is correct and contains training logs")


def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Run the dashboard
        os.environ["STREAMLIT_SERVER_PORT"] = str(args.port)
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        os.environ["STREAMLIT_SERVER_ADDRESS"] = args.host
        
        create_dashboard(args.log_dir, args.refresh_interval)
        
        return 0
    except Exception as e:
        logging.error(f"Dashboard failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    # When run directly, streamlit will handle the script execution
    # The script below is only for documentation
    print("To run the dashboard, use the following command:")
    print("streamlit run scripts/dashboard.py -- --log_dir=<log_dir>")
    
    # Attempt to run via streamlit if called directly
    if len(sys.argv) > 1 and sys.argv[1] != "--help":
        args = parse_args()
        print(f"Running dashboard with log_dir={args.log_dir}")
        os.system(f"streamlit run {__file__} -- "
                f"--log_dir={args.log_dir} "
                f"--port={args.port} "
                f"--host={args.host} "
                f"--refresh_interval={args.refresh_interval}")
    else:
        sys.exit(main()) 