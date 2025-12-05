"""Visualization functions using plotly and matplotlib."""

import pandas as pd
import numpy as np
from typing import Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Try plotly first, fallback to matplotlib
PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_reluctant_topics(
    topic_df: pd.DataFrame,
    top_n: int = 10,
    use_plotly: bool = True
):
    """
    Plot bar chart of reluctant topics (RQ1).
    
    Args:
        topic_df: DataFrame from rank_reluctant_topics
        top_n: Number of top topics to show
        use_plotly: Whether to use plotly (else matplotlib)
        
    Returns:
        Plotly figure or matplotlib figure
    """
    top_df = topic_df.head(top_n).copy()
    top_df = top_df.sort_values('final_rank', ascending=True)
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_df['final_rank'],
            y=[f"Topic {row['topic_id']}: {row['top_words'][:30]}" for _, row in top_df.iterrows()],
            orientation='h',
            text=[f"Reluctance: {row['reluctance_score']:.3f}" for _, row in top_df.iterrows()],
            textposition='auto'
        ))
        fig.update_layout(
            title="Top Topics by Reluctance to Respond",
            xaxis_title="Reluctance Score (Final Rank)",
            yaxis_title="Topic",
            height=400 + top_n * 30
        )
        return fig
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.5)))
        y_pos = np.arange(len(top_df))
        ax.barh(y_pos, top_df['final_rank'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Topic {row['topic_id']}" for _, row in top_df.iterrows()])
        ax.set_xlabel("Reluctance Score (Final Rank)")
        ax.set_title("Top Topics by Reluctance to Respond")
        plt.tight_layout()
        return fig


def plot_response_times_by_topic(
    df: pd.DataFrame,
    doc_topics: list,
    topic_ids: list,
    use_plotly: bool = True
):
    """
    Plot violin/box plot of response times by topic (RQ1).
    
    Args:
        df: DataFrame with response times
        doc_topics: List of topic distributions
        topic_ids: List of topic IDs to plot
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly figure or matplotlib figure
    """
    # Get messages with dominant topics
    # Ensure df and doc_topics are aligned
    df_aligned = df.reset_index(drop=True)
    topic_data = []
    for doc_idx, (_, row) in enumerate(df_aligned.iterrows()):
        if doc_idx < len(doc_topics):
            topic_probs = doc_topics[doc_idx]
            dominant_topic = np.argmax(topic_probs)
            if dominant_topic in topic_ids and pd.notna(row.get('response_time_min')):
                topic_data.append({
                    'topic_id': dominant_topic,
                    'response_time_min': row['response_time_min']
                })
    
    topic_df = pd.DataFrame(topic_data)
    
    if len(topic_df) == 0:
        return None
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure()
        for topic_id in topic_ids:
            topic_subset = topic_df[topic_df['topic_id'] == topic_id]['response_time_min']
            if len(topic_subset) > 0:
                fig.add_trace(go.Box(
                    y=topic_subset,
                    name=f"Topic {topic_id}",
                    boxmean='sd'
                ))
        fig.update_layout(
            title="Response Time Distribution by Topic",
            yaxis_title="Response Time (minutes)",
            xaxis_title="Topic"
        )
        return fig
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(figsize=(10, 6))
        data_to_plot = [topic_df[topic_df['topic_id'] == tid]['response_time_min'].values 
                        for tid in topic_ids]
        ax.boxplot(data_to_plot, labels=[f"Topic {tid}" for tid in topic_ids])
        ax.set_ylabel("Response Time (minutes)")
        ax.set_title("Response Time Distribution by Topic")
        plt.tight_layout()
        return fig


def plot_topic_prevalence_over_time(
    topic_over_time_df: pd.DataFrame,
    top_n_topics: int = None,
    use_plotly: bool = True
):
    """
    Plot topic prevalence over time (RQ2).
    
    Args:
        topic_over_time_df: DataFrame from topic_over_time
        top_n_topics: Number of top topics to show (None = all topics)
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly figure or matplotlib figure
    """
    if len(topic_over_time_df) == 0:
        return None
    
    # Get topics by average prevalence
    topic_avg = topic_over_time_df.groupby('topic_id')['prevalence'].mean().sort_values(ascending=False)
    
    if top_n_topics is not None:
        top_topics = topic_avg.head(top_n_topics).index.tolist()
    else:
        top_topics = topic_avg.index.tolist()  # All topics
    
    plot_df = topic_over_time_df[topic_over_time_df['topic_id'].isin(top_topics)].copy()
    plot_df['period'] = pd.to_datetime(plot_df['period'].astype(str))
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = px.line(
            plot_df,
            x='period',
            y='prevalence',
            color='topic_id',
            title="Topic Prevalence Over Time",
            labels={'prevalence': 'Topic Prevalence', 'period': 'Period', 'topic_id': 'Topic'}
        )
        return fig
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(figsize=(12, 6))
        for topic_id in top_topics:
            topic_data = plot_df[plot_df['topic_id'] == topic_id]
            ax.plot(topic_data['period'], topic_data['prevalence'], 
                   marker='o', label=f"Topic {topic_id}")
        ax.set_xlabel("Period")
        ax.set_ylabel("Topic Prevalence")
        ax.set_title("Topic Prevalence Over Time")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig


def plot_per_contact_topic_mix(
    contact_topic_df: pd.DataFrame,
    use_plotly: bool = True
):
    """
    Plot stacked bar chart of topic proportions per contact (RQ2).
    
    Args:
        contact_topic_df: DataFrame from get_per_contact_topic_mix
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly figure or matplotlib figure
    """
    if len(contact_topic_df) == 0:
        return None
    
    pivot_df = contact_topic_df.pivot(index='participant', columns='topic_id', values='proportion')
    pivot_df = pivot_df.fillna(0)
    
    # Create truncated labels for display (first 8 chars + '...' if longer)
    def truncate_label(label: str, max_len: int = 8) -> str:
        if len(label) > max_len:
            return label[:max_len] + '...'
        return label
    
    full_labels = [str(p) for p in pivot_df.index]
    truncated_labels = [truncate_label(p) for p in full_labels]
    
    # Use categorical positions for x-axis
    x_positions = list(range(len(pivot_df)))
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure()
        for topic_id in pivot_df.columns:
            fig.add_trace(go.Bar(
                name=f"Topic {topic_id}",
                x=x_positions,
                y=pivot_df[topic_id],
                customdata=full_labels,  # Full contact name in hover
                hovertemplate='<b>%{customdata}</b><br>Topic ' + str(topic_id) + ': %{y:.2%}<extra></extra>'
            ))
        fig.update_layout(
            barmode='stack',
            title="Topic Distribution by Contact",
            xaxis_title="Contact",
            yaxis_title="Topic Proportion",
            height=500,
            xaxis={
                'tickmode': 'array',
                'tickvals': x_positions,
                'ticktext': truncated_labels,
                'tickangle': -45
            }
        )
        return fig
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_df.plot(kind='bar', stacked=True, ax=ax)
        ax.set_xlabel("Contact")
        ax.set_ylabel("Topic Proportion")
        ax.set_title("Topic Distribution by Contact")
        ax.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(truncated_labels, rotation=45, ha='right')
        plt.tight_layout()
        return fig


def plot_group_vs_dm_response(
    stats: dict,
    use_plotly: bool = True
):
    """
    Plot response statistics by group size (RQ3).
    
    Args:
        stats: Dictionary from group_vs_dm_stats
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly figure or matplotlib figure
    """
    by_category = stats.get('by_category', pd.DataFrame())
    
    if len(by_category) == 0:
        return None
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Reply Rate by Group Size', 'Median Response Time by Group Size')
        )
        
        fig.add_trace(
            go.Bar(x=by_category['category'], y=by_category['reply_rate'],
                  name='Reply Rate'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=by_category['category'], y=by_category['median_response_time'],
                  name='Median Response Time (min)'),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Group Category", row=1, col=1)
        fig.update_xaxes(title_text="Group Category", row=1, col=2)
        fig.update_yaxes(title_text="Reply Rate", row=1, col=1)
        fig.update_yaxes(title_text="Response Time (minutes)", row=1, col=2)
        fig.update_layout(title_text="Group vs One-to-One Responsiveness", height=400)
        
        return fig
    else:
        # Matplotlib fallback
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.bar(by_category['category'], by_category['reply_rate'])
        ax1.set_xlabel("Group Category")
        ax1.set_ylabel("Reply Rate")
        ax1.set_title("Reply Rate by Group Size")
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(by_category['category'], by_category['median_response_time'])
        ax2.set_xlabel("Group Category")
        ax2.set_ylabel("Median Response Time (minutes)")
        ax2.set_title("Median Response Time by Group Size")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig


def plot_response_time_boxplot(
    df: pd.DataFrame,
    use_plotly: bool = True
):
    """
    Plot box plot of response times by group size category.
    
    Args:
        df: DataFrame with response_time_min and group_size (or group_category)
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly figure or matplotlib figure
    """
    df = df[df['response_time_min'].notna()].copy()
    
    if len(df) == 0:
        return None
    
    # Create group_category if it doesn't exist
    if 'group_category' not in df.columns and 'group_size' in df.columns:
        def categorize_group_size(size):
            if pd.isna(size) or size < 2:
                return "Unknown"
            elif size == 2:
                return "One-to-One"
            elif size <= 4:
                return "Small (3-4)"
            elif size <= 8:
                return "Medium (5-8)"
            else:
                return "Large (9+)"
        df['group_category'] = df['group_size'].apply(categorize_group_size)
    elif 'group_category' not in df.columns:
        # If neither exists, create a default category
        df['group_category'] = "Unknown"
    
    if use_plotly and PLOTLY_AVAILABLE:
        categories = df['group_category'].unique()
        fig = go.Figure()
        for cat in categories:
            cat_data = df[df['group_category'] == cat]['response_time_min']
            fig.add_trace(go.Box(y=cat_data, name=cat))
        fig.update_layout(
            title="Response Time Distribution by Group Size",
            yaxis_title="Response Time (minutes)",
            xaxis_title="Group Category"
        )
        return fig
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = df['group_category'].unique()
        data_to_plot = [df[df['group_category'] == cat]['response_time_min'].values 
                        for cat in categories]
        ax.boxplot(data_to_plot, labels=categories)
        ax.set_ylabel("Response Time (minutes)")
        ax.set_xlabel("Group Category")
        ax.set_title("Response Time Distribution by Group Size")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig


def plot_daily_response_times(
    daily_df: pd.DataFrame,
    use_plotly: bool = True
):
    """
    Plot average response time per day over time.
    
    Args:
        daily_df: DataFrame from compute_daily_response_times with columns: date, avg_response_time_min, count, and optionally message_details
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly figure or matplotlib figure
    """
    if len(daily_df) == 0:
        return None
    
    daily_df = daily_df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        # Check if message_details column exists
        has_details = 'message_details' in daily_df.columns
        
        if has_details:
            # Create custom hover text with message details
            def format_hover_text(row):
                base_text = f"Date: {row['date'].strftime('%Y-%m-%d')}<br>Avg Response Time: {row['avg_response_time_min']:.1f} minutes<br>Count: {row['count']}<br><br>"
                
                if row['message_details'] and len(row['message_details']) > 0:
                    base_text += "<b>Sample messages:</b><br>"
                    for i, detail in enumerate(row['message_details'][:5], 1):  # Limit to 5
                        text_preview = detail['text'].replace('<', '&lt;').replace('>', '&gt;')
                        if len(text_preview) > 80:
                            text_preview = text_preview[:80] + "..."
                        base_text += f"{i}. <b>{detail['participant']}</b> ({detail['response_time']:.1f} min): {text_preview}<br>"
                
                return base_text
            
            hover_texts = daily_df.apply(format_hover_text, axis=1)
            customdata = daily_df['count']
        else:
            hover_texts = None
            customdata = daily_df['count']
        
        if hover_texts is not None:
            hovertemplate = '%{text}<extra></extra>'
            fig.add_trace(go.Scatter(
                x=daily_df['date'],
                y=daily_df['avg_response_time_min'],
                mode='lines+markers',
                name='Avg Response Time',
                text=hover_texts,
                hovertemplate=hovertemplate,
                customdata=customdata
            ))
        else:
            fig.add_trace(go.Scatter(
                x=daily_df['date'],
                y=daily_df['avg_response_time_min'],
                mode='lines+markers',
                name='Avg Response Time',
                hovertemplate='Date: %{x}<br>Avg Response Time: %{y:.1f} minutes<br>Count: %{customdata}<extra></extra>',
                customdata=customdata
            ))
        
        fig.update_layout(
            title="Average Response Time Per Day Over Time",
            xaxis_title="Date",
            yaxis_title="Average Response Time (minutes)",
            hovermode='closest',
            height=400
        )
        return fig
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(daily_df['date'], daily_df['avg_response_time_min'], marker='o', linewidth=2, markersize=4)
        ax.set_xlabel("Date")
        ax.set_ylabel("Average Response Time (minutes)")
        ax.set_title("Average Response Time Per Day Over Time")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


def plot_daily_text_length(
    daily_df: pd.DataFrame,
    use_plotly: bool = True
):
    """
    Plot average text length in words per day over time.
    
    Args:
        daily_df: DataFrame from compute_daily_text_length with columns: date, avg_word_count, count
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly figure or matplotlib figure
    """
    if len(daily_df) == 0:
        return None
    
    daily_df = daily_df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_df['date'],
            y=daily_df['avg_word_count'],
            mode='lines+markers',
            name='Avg Word Count',
            hovertemplate='Date: %{x}<br>Avg Word Count: %{y:.1f} words<br>Count: %{customdata}<extra></extra>',
            customdata=daily_df['count']
        ))
        fig.update_layout(
            title="Average Text Length (Words) Per Day Over Time",
            xaxis_title="Date",
            yaxis_title="Average Word Count",
            hovermode='x unified',
            height=400
        )
        return fig
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(daily_df['date'], daily_df['avg_word_count'], marker='o', linewidth=2, markersize=4)
        ax.set_xlabel("Date")
        ax.set_ylabel("Average Word Count")
        ax.set_title("Average Text Length (Words) Per Day Over Time")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


def plot_daily_sent_count(
    daily_df: pd.DataFrame,
    use_plotly: bool = True,
    title: str = "Number of Texts Sent Per Day Over Time",
    yaxis_label: str = "Number of Messages Sent"
):
    """
    Plot number of texts sent/received per day over time.
    
    Args:
        daily_df: DataFrame with columns: date, sent_count
        use_plotly: Whether to use plotly
        title: Plot title (customizable)
        yaxis_label: Y-axis label (customizable)
        
    Returns:
        Plotly figure or matplotlib figure
    """
    if len(daily_df) == 0:
        return None
    
    daily_df = daily_df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily_df['date'],
            y=daily_df['sent_count'],
            name=yaxis_label,
            hovertemplate=f'Date: %{{x}}<br>{yaxis_label}: %{{y}}<extra></extra>'
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=yaxis_label,
            hovermode='x unified',
            height=400
        )
        return fig
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(daily_df['date'], daily_df['sent_count'], width=1.0, alpha=0.7)
        ax.set_xlabel("Date")
        ax.set_ylabel(yaxis_label)
        ax.set_title(title)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig


def plot_contact_reluctant_topic_count(
    contact_stats_df: pd.DataFrame,
    top_n: int = 7,
    use_plotly: bool = True
):
    """
    Plot contacts by number of messages with high prevalence of reluctant topics.
    
    Args:
        contact_stats_df: DataFrame from get_contact_reluctant_topic_stats
        top_n: Number of top contacts to show
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly figure or matplotlib figure
    """
    if len(contact_stats_df) == 0:
        return None
    
    plot_df = contact_stats_df.head(top_n).copy()
    plot_df = plot_df.sort_values('high_reluctant_topic_count', ascending=False).reset_index(drop=True)
    
    # Create truncated labels for display (first 8 chars + '...' if longer)
    def truncate_label(label: str, max_len: int = 8) -> str:
        if len(label) > max_len:
            return label[:max_len] + '...'
        return label
    
    truncated_labels = [truncate_label(str(c)) for c in plot_df['contact']]
    full_labels = [str(c) for c in plot_df['contact']]
    
    # Use categorical positions for x-axis
    x_positions = list(range(len(plot_df)))
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_positions,
            y=plot_df['high_reluctant_topic_count'],
            text=plot_df['high_reluctant_topic_count'],
            textposition='auto',
            hovertemplate='<b>%{customdata}</b><br>High Reluctant Topic Messages: %{y}<extra></extra>',
            customdata=full_labels  # Full contact name in hover
        ))
        fig.update_layout(
            title="Contacts by Number of High Reluctant Topic Messages",
            xaxis_title="Contact",
            yaxis_title="Number of Messages with High Reluctant Topic Prevalence",
            height=400,
            xaxis={
                'tickmode': 'array',
                'tickvals': x_positions,
                'ticktext': truncated_labels,
                'tickangle': -45
            }
        )
        return fig
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x_positions, plot_df['high_reluctant_topic_count'])
        ax.set_xlabel("Contact")
        ax.set_ylabel("Number of Messages with High Reluctant Topic Prevalence")
        ax.set_title("Contacts by Number of High Reluctant Topic Messages")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(truncated_labels, rotation=45, ha='right')
        
        # Add full labels as tooltips/annotations (show on hover in interactive mode)
        for i, (bar, full_label) in enumerate(zip(bars, full_labels)):
            if full_label != truncated_labels[i]:  # Only annotate if truncated
                bar.set_label(full_label)
        
        plt.tight_layout()
        return fig


def plot_contact_avg_reluctance(
    contact_stats_df: pd.DataFrame,
    top_n: int = 7,
    use_plotly: bool = True
):
    """
    Plot contacts by average reluctance score.
    
    Args:
        contact_stats_df: DataFrame from get_contact_reluctant_topic_stats
        top_n: Number of top contacts to show
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly figure or matplotlib figure
    """
    if len(contact_stats_df) == 0:
        return None
    
    # Sort by average reluctance score (highest first)
    plot_df = contact_stats_df.nlargest(top_n, 'avg_reluctance_score').copy()
    plot_df = plot_df.sort_values('avg_reluctance_score', ascending=False).reset_index(drop=True)
    
    # Create truncated labels for display (first 8 chars + '...' if longer)
    def truncate_label(label: str, max_len: int = 8) -> str:
        if len(label) > max_len:
            return label[:max_len] + '...'
        return label
    
    truncated_labels = [truncate_label(str(c)) for c in plot_df['contact']]
    full_labels = [str(c) for c in plot_df['contact']]
    
    # Use categorical positions for x-axis
    x_positions = list(range(len(plot_df)))
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_positions,
            y=plot_df['avg_reluctance_score'],
            text=[f"{x:.3f}" for x in plot_df['avg_reluctance_score']],
            textposition='auto',
            hovertemplate='<b>%{customdata[0]}</b><br>Avg Reluctance Score: %{y:.3f}<br>Total Messages: %{customdata[1]}<extra></extra>',
            customdata=list(zip(full_labels, plot_df['total_messages']))  # Full contact name + total messages in hover
        ))
        fig.update_layout(
            title="Contacts by Average Reluctance Score",
            xaxis_title="Contact",
            yaxis_title="Average Reluctance Score",
            height=400,
            xaxis={
                'tickmode': 'array',
                'tickvals': x_positions,
                'ticktext': truncated_labels,
                'tickangle': -45
            }
        )
        return fig
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x_positions, plot_df['avg_reluctance_score'])
        ax.set_xlabel("Contact")
        ax.set_ylabel("Average Reluctance Score")
        ax.set_title("Contacts by Average Reluctance Score")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(truncated_labels, rotation=45, ha='right')
        
        # Add full labels as tooltips/annotations (show on hover in interactive mode)
        for i, (bar, full_label) in enumerate(zip(bars, full_labels)):
            if full_label != truncated_labels[i]:  # Only annotate if truncated
                bar.set_label(full_label)
        
        plt.tight_layout()
        return fig


def plot_contact_reluctance_proportions(df: pd.DataFrame, top_n: int = 10, use_plotly: bool = True):
    """Plot contacts by proportion of high reluctance messages (RQ1 enhancement)."""
    if df is None or len(df) == 0:
        return None
    
    # Take top N contacts
    plot_df = df.head(top_n).copy()
    
    # Handle long names
    full_labels = plot_df['contact'].tolist()
    truncated_labels = [label[:8] + '...' if len(label) > 8 else label for label in full_labels]
    x_positions = list(range(len(plot_df)))
    
    if use_plotly:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[
            go.Bar(
                x=x_positions,
                y=plot_df['high_reluctance_proportion'],
                text=[f"{p:.1%}" for p in plot_df['high_reluctance_proportion']],
                textposition='auto',
                marker_color='indianred',
                customdata=[[full, total, count] for full, total, count in 
                           zip(full_labels, plot_df['total_messages'], plot_df['high_reluctance_count'])],
                hovertemplate='<b>%{customdata[0]}</b><br>' +
                             'Proportion: %{y:.1%}<br>' +
                             'High Reluctance: %{customdata[2]}<br>' +
                             'Total Messages: %{customdata[1]}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Contacts by Proportion of High Reluctance Messages",
            xaxis_title="Contact",
            yaxis_title="Proportion",
            showlegend=False,
            height=400,
            xaxis={
                'tickmode': 'array',
                'tickvals': x_positions,
                'ticktext': truncated_labels,
                'tickangle': -45
            },
            yaxis={'tickformat': '.0%'}
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x_positions, plot_df['high_reluctance_proportion'], color='indianred')
        ax.set_xlabel("Contact")
        ax.set_ylabel("Proportion")
        ax.set_title("Contacts by Proportion of High Reluctance Messages")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(truncated_labels, rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.tight_layout()
        return fig


def plot_conversation_starters(df: pd.DataFrame, model, dictionary, top_n: int = 10, use_plotly: bool = True):
    """Plot conversation starter topics (RQ4)."""
    if df is None or len(df) == 0:
        return None
    
    from topics import get_topic_words
    
    # Get top N topics
    plot_df = df.head(top_n).copy()
    
    # Get topic words
    topic_labels = []
    for topic_id in plot_df['topic_id']:
        try:
            words = get_topic_words(model, dictionary, topic_id, num_words=3)
            label = ', '.join([w for w, _ in words])
        except:
            label = f"Topic {topic_id}"
        topic_labels.append(label)
    
    if use_plotly:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(range(len(plot_df))),
                y=plot_df['starter_score'],
                text=[f"{s:.3f}" for s in plot_df['starter_score']],
                textposition='auto',
                marker_color='lightseagreen',
                customdata=[[label, rr, rt, sp] for label, rr, rt, sp in 
                           zip(topic_labels, plot_df['reply_rate'], 
                               plot_df['avg_response_time'], plot_df['starter_probability'])],
                hovertemplate='<b>%{customdata[0]}</b><br>' +
                             'Starter Score: %{y:.3f}<br>' +
                             'Reply Rate: %{customdata[1]:.1%}<br>' +
                             'Avg Response: %{customdata[2]:.1f} min<br>' +
                             'Starter Prob: %{customdata[3]:.1%}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Top Conversation Starter Topics",
            xaxis_title="Topic",
            yaxis_title="Starter Score",
            showlegend=False,
            height=500,
            xaxis={
                'tickmode': 'array',
                'tickvals': list(range(len(plot_df))),
                'ticktext': [f"T{tid}" for tid in plot_df['topic_id']],
                'tickangle': -45
            }
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(plot_df)), plot_df['starter_score'], color='lightseagreen')
        ax.set_xlabel("Topic")
        ax.set_ylabel("Starter Score")
        ax.set_title("Top Conversation Starter Topics")
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels([f"T{tid}" for tid in plot_df['topic_id']], rotation=45, ha='right')
        plt.tight_layout()
        return fig


def plot_conversation_enders(df: pd.DataFrame, model, dictionary, top_n: int = 10, use_plotly: bool = True):
    """Plot conversation ender topics (RQ5)."""
    if df is None or len(df) == 0:
        return None
    
    from topics import get_topic_words
    
    plot_df = df.head(top_n).copy()
    
    topic_labels = []
    for topic_id in plot_df['topic_id']:
        try:
            words = get_topic_words(model, dictionary, topic_id, num_words=3)
            label = ', '.join([w for w, _ in words])
        except:
            label = f"Topic {topic_id}"
        topic_labels.append(label)
    
    if use_plotly:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(range(len(plot_df))),
                y=plot_df['ender_score'],
                text=[f"{s:.3f}" for s in plot_df['ender_score']],
                textposition='auto',
                marker_color='coral',
                customdata=[[label, nrr, rt, ep] for label, nrr, rt, ep in 
                           zip(topic_labels, plot_df['no_reply_rate'], 
                               plot_df['avg_response_time'], plot_df['ender_probability'])],
                hovertemplate='<b>%{customdata[0]}</b><br>' +
                             'Ender Score: %{y:.3f}<br>' +
                             'No Reply Rate: %{customdata[1]:.1%}<br>' +
                             'Avg Response: %{customdata[2]:.1f} min<br>' +
                             'Ender Prob: %{customdata[3]:.1%}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Top Conversation Ender Topics",
            xaxis_title="Topic",
            yaxis_title="Ender Score",
            showlegend=False,
            height=500,
            xaxis={
                'tickmode': 'array',
                'tickvals': list(range(len(plot_df))),
                'ticktext': [f"T{tid}" for tid in plot_df['topic_id']],
                'tickangle': -45
            }
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(plot_df)), plot_df['ender_score'], color='coral')
        ax.set_xlabel("Topic")
        ax.set_ylabel("Ender Score")
        ax.set_title("Top Conversation Ender Topics")
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels([f"T{tid}" for tid in plot_df['topic_id']], rotation=45, ha='right')
        plt.tight_layout()
        return fig


def plot_topics_by_closeness(df: pd.DataFrame, model, dictionary, top_n: int = 10, use_plotly: bool = True):
    """Plot topics by close vs. acquaintance contacts (RQ6)."""
    if df is None or len(df) == 0:
        return None
    
    from topics import get_topic_words
    
    # Get top N topics by odds ratio difference from 1
    df = df.copy()
    df['odds_diff'] = np.abs(np.log(df['odds_ratio']))
    plot_df = df.nlargest(top_n, 'odds_diff').copy()
    
    topic_labels = []
    for topic_id in plot_df['topic_id']:
        try:
            words = get_topic_words(model, dictionary, topic_id, num_words=3)
            label = ', '.join([w for w, _ in words])
        except:
            label = f"Topic {topic_id}"
        topic_labels.append(label)
    
    if use_plotly:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(range(len(plot_df))),
                y=plot_df['odds_ratio'],
                text=[f"{o:.2f}x" for o in plot_df['odds_ratio']],
                textposition='auto',
                marker_color=[
                    'steelblue' if cat == 'Close contacts' else 
                    'lightcoral' if cat == 'Acquaintances' else 'lightgray'
                    for cat in plot_df['category']
                ],
                customdata=[[label, cat, cf, af] for label, cat, cf, af in 
                           zip(topic_labels, plot_df['category'], 
                               plot_df['close_frequency'], plot_df['acquaintance_frequency'])],
                hovertemplate='<b>%{customdata[0]}</b><br>' +
                             'Odds Ratio: %{y:.2f}x<br>' +
                             'Category: %{customdata[1]}<br>' +
                             'Close Freq: %{customdata[2]:.1%}<br>' +
                             'Acquaint Freq: %{customdata[3]:.1%}<extra></extra>'
            )
        ])
        
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                      annotation_text="Equal prevalence")
        
        fig.update_layout(
            title="Topics by Closeness (Odds Ratio: Close / Acquaintance)",
            xaxis_title="Topic",
            yaxis_title="Odds Ratio",
            showlegend=False,
            height=500,
            xaxis={
                'tickmode': 'array',
                'tickvals': list(range(len(plot_df))),
                'ticktext': [f"T{tid}" for tid in plot_df['topic_id']],
                'tickangle': -45
            }
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['steelblue' if cat == 'Close contacts' else 
                 'lightcoral' if cat == 'Acquaintances' else 'lightgray'
                 for cat in plot_df['category']]
        ax.bar(range(len(plot_df)), plot_df['odds_ratio'], color=colors)
        ax.axhline(y=1.0, linestyle='--', color='gray', alpha=0.7)
        ax.set_xlabel("Topic")
        ax.set_ylabel("Odds Ratio")
        ax.set_title("Topics by Closeness")
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels([f"T{tid}" for tid in plot_df['topic_id']], rotation=45, ha='right')
        plt.tight_layout()
        return fig


def plot_topics_by_time_heatmap(df: pd.DataFrame, model, dictionary, use_plotly: bool = True):
    """Plot heatmap of topics by time of day (RQ7)."""
    if df is None or len(df) == 0:
        return None
    
    from topics import get_topic_words
    
    # Get topic labels
    topic_labels = []
    for topic_id in df['topic_id']:
        try:
            words = get_topic_words(model, dictionary, topic_id, num_words=3)
            label = f"T{topic_id}: {', '.join([w for w, _ in words[:2]])}"
        except:
            label = f"Topic {topic_id}"
        topic_labels.append(label)
    
    # Prepare data matrix
    time_periods = ['morning', 'afternoon', 'evening', 'night']
    data_matrix = df[time_periods].values
    
    if use_plotly:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=['Morning', 'Afternoon', 'Evening', 'Night'],
            y=topic_labels,
            colorscale='YlOrRd',
            hovertemplate='Topic: %{y}<br>Time: %{x}<br>Frequency: %{z:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Topics by Time of Day",
            xaxis_title="Time Period",
            yaxis_title="Topic",
            height=max(400, len(df) * 20)
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.3)))
        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(time_periods)))
        ax.set_xticklabels(['Morning', 'Afternoon', 'Evening', 'Night'])
        ax.set_yticks(range(len(topic_labels)))
        ax.set_yticklabels(topic_labels, fontsize=8)
        ax.set_xlabel("Time Period")
        ax.set_ylabel("Topic")
        ax.set_title("Topics by Time of Day")
        plt.colorbar(im, ax=ax, label='Frequency')
        plt.tight_layout()
        return fig


def plot_high_response_topics(df: pd.DataFrame, model, dictionary, top_n: int = 10, use_plotly: bool = True):
    """Plot topics with highest response likelihood (RQ8)."""
    if df is None or len(df) == 0:
        return None
    
    from topics import get_topic_words
    
    plot_df = df.head(top_n).copy()
    
    topic_labels = []
    for topic_id in plot_df['topic_id']:
        try:
            words = get_topic_words(model, dictionary, topic_id, num_words=3)
            label = ', '.join([w for w, _ in words])
        except:
            label = f"Topic {topic_id}"
        topic_labels.append(label)
    
    if use_plotly:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(range(len(plot_df))),
                y=plot_df['reply_rate'],
                text=[f"{r:.1%}" for r in plot_df['reply_rate']],
                textposition='auto',
                marker_color='mediumseagreen',
                customdata=[[label, rt, mc] for label, rt, mc in 
                           zip(topic_labels, plot_df['avg_response_time'], plot_df['message_count'])],
                hovertemplate='<b>%{customdata[0]}</b><br>' +
                             'Reply Rate: %{y:.1%}<br>' +
                             'Avg Response: %{customdata[1]:.1f} min<br>' +
                             'Messages: %{customdata[2]}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Topics You're Most Likely to Respond To",
            xaxis_title="Topic",
            yaxis_title="Reply Rate",
            showlegend=False,
            height=500,
            xaxis={
                'tickmode': 'array',
                'tickvals': list(range(len(plot_df))),
                'ticktext': [f"T{tid}" for tid in plot_df['topic_id']],
                'tickangle': -45
            },
            yaxis={'tickformat': '.0%'}
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(plot_df)), plot_df['reply_rate'], color='mediumseagreen')
        ax.set_xlabel("Topic")
        ax.set_ylabel("Reply Rate")
        ax.set_title("Topics You're Most Likely to Respond To")
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels([f"T{tid}" for tid in plot_df['topic_id']], rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.tight_layout()
        return fig


def plot_reluctant_embedding_topics(df: pd.DataFrame, top_n: int = 15, use_plotly: bool = True):
    """
    Plot top reluctant topics from embedding-based model (RQ8).
    
    Args:
        df: DataFrame from rank_reluctant_topics_embedding
        top_n: Number of top topics to show
        use_plotly: Whether to use Plotly (True) or Matplotlib (False)
        
    Returns:
        Plotly or Matplotlib figure
    """
    if df is None or len(df) == 0:
        return None
    
    plot_df = df.head(top_n).copy()
    
    if use_plotly:
        import plotly.graph_objects as go
        
        # Truncate topic words for display
        truncated_labels = [label[:30] + '...' if len(label) > 30 else label 
                           for label in plot_df['top_words']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(range(len(plot_df))),
                y=plot_df['final_rank'],
                text=[f"{r:.3f}" for r in plot_df['final_rank']],
                textposition='auto',
                marker_color='indianred',
                customdata=[[tid, words, reluc, freq] for tid, words, reluc, freq in 
                           zip(plot_df['topic_id'], plot_df['top_words'], 
                               plot_df['reluctance_score'], plot_df['frequency'])],
                hovertemplate='<b>Topic %{customdata[0]}</b><br>' +
                             'Words: %{customdata[1]}<br>' +
                             'Rank Score: %{y:.3f}<br>' +
                             'Avg Reluctance: %{customdata[2]:.3f}<br>' +
                             'Frequency: %{customdata[3]}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Fine-Grained Topics You Tend Not to Engage With (Embedding-Based)",
            xaxis_title="Topic",
            yaxis_title="Rank Score (Reluctance × log(1 + Frequency))",
            showlegend=False,
            height=500,
            xaxis={
                'tickmode': 'array',
                'tickvals': list(range(len(plot_df))),
                'ticktext': [f"T{tid}" for tid in plot_df['topic_id']],
                'tickangle': -45
            }
        )
        return fig
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(range(len(plot_df)), plot_df['final_rank'], color='indianred')
        ax.set_xlabel("Topic")
        ax.set_ylabel("Rank Score")
        ax.set_title("Fine-Grained Topics You Tend Not to Engage With (Embedding-Based)")
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels([f"T{tid}" for tid in plot_df['topic_id']], rotation=45, ha='right')
        plt.tight_layout()
        return fig


def plot_daily_total_count(df: pd.DataFrame, use_plotly: bool = True):
    """Plot total messages (inbound + outbound) per day."""
    if df is None or len(df) == 0:
        return None
    
    if use_plotly and PLOTLY_AVAILABLE:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['total_count'],
            mode='lines+markers',
            name='Total Messages',
            line=dict(color='royalblue', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title="Total Messages Per Day (Inbound + Outbound)",
            xaxis_title="Date",
            yaxis_title="Total Messages",
            hovermode='x unified',
            height=400
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['date'], df['total_count'], marker='o', linewidth=2, color='royalblue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Messages")
        ax.set_title("Total Messages Per Day (Inbound + Outbound)")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig


# ============================================================================
# RQ5: Sentiment Analysis Visualizations
# ============================================================================

def plot_sentiment_distribution(summary: dict, use_plotly: bool = True):
    """
    Plot sentiment distribution as pie/bar charts.
    
    Args:
        summary: Dictionary from global_sentiment_summary()
        use_plotly: Whether to use plotly (True) or matplotlib (False)
    """
    if use_plotly and PLOTLY_AVAILABLE:
        from plotly.subplots import make_subplots
        
        # Create subplots: overall, sent, received
        specs = [[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]]
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Overall", "Sent Messages", "Received Messages"),
            specs=specs
        )
        
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        
        # Overall
        if 'overall' in summary:
            fig.add_trace(go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[
                    summary['overall']['pct_positive'],
                    summary['overall']['pct_negative'],
                    summary['overall']['pct_neutral']
                ],
                marker=dict(colors=[colors['positive'], colors['negative'], colors['neutral']]),
                name="Overall"
            ), row=1, col=1)
        
        # Sent (out)
        if 'out' in summary:
            fig.add_trace(go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[
                    summary['out']['pct_positive'],
                    summary['out']['pct_negative'],
                    summary['out']['pct_neutral']
                ],
                marker=dict(colors=[colors['positive'], colors['negative'], colors['neutral']]),
                name="Sent"
            ), row=1, col=2)
        
        # Received (in)
        if 'in' in summary:
            fig.add_trace(go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[
                    summary['in']['pct_positive'],
                    summary['in']['pct_negative'],
                    summary['in']['pct_neutral']
                ],
                marker=dict(colors=[colors['positive'], colors['negative'], colors['neutral']]),
                name="Received"
            ), row=1, col=3)
        
        fig.update_layout(
            title="Sentiment Distribution",
            height=400,
            showlegend=True
        )
        
        return fig
    else:
        # Matplotlib fallback
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors_list = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        for idx, (key, title) in enumerate([('overall', 'Overall'), 
                                             ('out', 'Sent Messages'), 
                                             ('in', 'Received Messages')]):
            if key in summary:
                axes[idx].pie(
                    [summary[key]['pct_positive'], 
                     summary[key]['pct_negative'], 
                     summary[key]['pct_neutral']],
                    labels=['Positive', 'Negative', 'Neutral'],
                    colors=colors_list,
                    autopct='%1.1f%%'
                )
                axes[idx].set_title(title)
        
        plt.tight_layout()
        return fig


def plot_sentiment_over_time(time_df: pd.DataFrame, use_plotly: bool = True):
    """
    Plot sentiment trends over time.
    
    Args:
        time_df: DataFrame from sentiment_over_time()
        use_plotly: Whether to use plotly (True) or matplotlib (False)
    """
    if len(time_df) == 0:
        return None
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Average Sentiment Score Over Time", 
                           "Sentiment Distribution Over Time"),
            vertical_spacing=0.15
        )
        
        # Average sentiment score
        fig.add_trace(go.Scatter(
            x=time_df['period'],
            y=time_df['avg_sentiment'],
            mode='lines+markers',
            name='Overall',
            line=dict(color='purple', width=2),
            marker=dict(size=6)
        ), row=1, col=1)
        
        # Add sent/received if available
        if 'avg_sentiment_out' in time_df.columns:
            fig.add_trace(go.Scatter(
                x=time_df['period'],
                y=time_df['avg_sentiment_out'],
                mode='lines+markers',
                name='Sent',
                line=dict(color='blue', width=2, dash='dash'),
                marker=dict(size=4)
            ), row=1, col=1)
        
        if 'avg_sentiment_in' in time_df.columns:
            fig.add_trace(go.Scatter(
                x=time_df['period'],
                y=time_df['avg_sentiment_in'],
                mode='lines+markers',
                name='Received',
                line=dict(color='orange', width=2, dash='dash'),
                marker=dict(size=4)
            ), row=1, col=1)
        
        # Sentiment distribution percentages
        fig.add_trace(go.Scatter(
            x=time_df['period'],
            y=time_df['pct_positive'],
            mode='lines',
            name='% Positive',
            line=dict(color='#2ecc71', width=2),
            stackgroup='one'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=time_df['period'],
            y=time_df['pct_neutral'],
            mode='lines',
            name='% Neutral',
            line=dict(color='#95a5a6', width=2),
            stackgroup='one'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=time_df['period'],
            y=time_df['pct_negative'],
            mode='lines',
            name='% Negative',
            line=dict(color='#e74c3c', width=2),
            stackgroup='one'
        ), row=2, col=1)
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
        fig.update_yaxes(title_text="Percentage", row=2, col=1)
        
        fig.update_layout(
            height=800,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    else:
        # Matplotlib fallback
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Average sentiment
        axes[0].plot(time_df['period'], time_df['avg_sentiment'], 
                    marker='o', linewidth=2, label='Overall', color='purple')
        
        if 'avg_sentiment_out' in time_df.columns:
            axes[0].plot(time_df['period'], time_df['avg_sentiment_out'], 
                        marker='s', linewidth=2, linestyle='--', label='Sent', color='blue')
        
        if 'avg_sentiment_in' in time_df.columns:
            axes[0].plot(time_df['period'], time_df['avg_sentiment_in'], 
                        marker='^', linewidth=2, linestyle='--', label='Received', color='orange')
        
        axes[0].set_ylabel('Sentiment Score')
        axes[0].set_title('Average Sentiment Score Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        # Sentiment distribution
        axes[1].fill_between(time_df['period'], 0, time_df['pct_positive'], 
                            label='Positive', color='#2ecc71', alpha=0.7)
        axes[1].fill_between(time_df['period'], time_df['pct_positive'], 
                            time_df['pct_positive'] + time_df['pct_neutral'],
                            label='Neutral', color='#95a5a6', alpha=0.7)
        axes[1].fill_between(time_df['period'], 
                            time_df['pct_positive'] + time_df['pct_neutral'],
                            100,
                            label='Negative', color='#e74c3c', alpha=0.7)
        
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Percentage')
        axes[1].set_title('Sentiment Distribution Over Time')
        axes[1].legend()
        axes[1].set_ylim(0, 100)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig


def plot_chat_sentiment_over_time(time_df: pd.DataFrame, chat_name: str, use_plotly: bool = True):
    """
    Plot sentiment over time for a specific chat.
    
    Args:
        time_df: DataFrame from per_chat_sentiment()['time_series']
        chat_name: Name of the chat for title
        use_plotly: Whether to use plotly (True) or matplotlib (False)
    """
    if len(time_df) == 0:
        return None
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        # Overall sentiment
        fig.add_trace(go.Scatter(
            x=time_df['period'],
            y=time_df['avg_sentiment'],
            mode='lines+markers',
            name='Overall',
            line=dict(color='purple', width=3),
            marker=dict(size=6)
        ))
        
        # Sent/received if available
        if 'avg_sentiment_out' in time_df.columns:
            # Filter out NaN values for sent
            sent_mask = ~time_df['avg_sentiment_out'].isna()
            fig.add_trace(go.Scatter(
                x=time_df.loc[sent_mask, 'period'],
                y=time_df.loc[sent_mask, 'avg_sentiment_out'],
                mode='lines+markers',
                name='Sent',
                line=dict(color='blue', width=2, dash='dash'),
                marker=dict(size=4)
            ))
        
        if 'avg_sentiment_in' in time_df.columns:
            # Filter out NaN values for received
            recv_mask = ~time_df['avg_sentiment_in'].isna()
            fig.add_trace(go.Scatter(
                x=time_df.loc[recv_mask, 'period'],
                y=time_df.loc[recv_mask, 'avg_sentiment_in'],
                mode='lines+markers',
                name='Received',
                line=dict(color='orange', width=2, dash='dash'),
                marker=dict(size=4)
            ))
        
        fig.add_hline(y=0, line=dict(color='gray', dash='dot', width=1), 
                     annotation_text="Neutral")
        
        fig.update_layout(
            title=f"Sentiment Over Time: {chat_name}",
            xaxis_title="Time",
            yaxis_title="Sentiment Score",
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        return fig
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(time_df['period'], time_df['avg_sentiment'], 
               marker='o', linewidth=2, label='Overall', color='purple')
        
        if 'avg_sentiment_out' in time_df.columns:
            sent_mask = ~time_df['avg_sentiment_out'].isna()
            ax.plot(time_df.loc[sent_mask, 'period'], 
                   time_df.loc[sent_mask, 'avg_sentiment_out'], 
                   marker='s', linewidth=2, linestyle='--', label='Sent', color='blue')
        
        if 'avg_sentiment_in' in time_df.columns:
            recv_mask = ~time_df['avg_sentiment_in'].isna()
            ax.plot(time_df.loc[recv_mask, 'period'], 
                   time_df.loc[recv_mask, 'avg_sentiment_in'], 
                   marker='^', linewidth=2, linestyle='--', label='Received', color='orange')
        
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Neutral')
        ax.set_xlabel('Time')
        ax.set_ylabel('Sentiment Score')
        ax.set_title(f'Sentiment Over Time: {chat_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig


def plot_top_contacts_by_positive_sentiment(
    contact_stats: pd.DataFrame,
    metric_type: str = 'overall',
    top_n: int = 10,
    use_plotly: bool = True
):
    """
    Plot top contacts by proportion of positive sentiment messages.
    
    Args:
        contact_stats: DataFrame from per_contact_sentiment_stats()
        metric_type: 'overall', 'sent', or 'received'
        top_n: Number of top contacts to show
        use_plotly: Whether to use plotly (True) or matplotlib (False)
    """
    if len(contact_stats) == 0:
        return None
    
    # Determine which column to use
    if metric_type == 'sent':
        pct_col = 'pct_positive_sent'
        total_col = 'sent_total'
        title = "Top Contacts by Positive Sentiment (Sent Messages)"
    elif metric_type == 'received':
        pct_col = 'pct_positive_received'
        total_col = 'received_total'
        title = "Top Contacts by Positive Sentiment (Received Messages)"
    else:  # overall
        pct_col = 'pct_positive_overall'
        total_col = 'total_messages'
        title = "Top Contacts by Positive Sentiment (All Messages)"
    
    # Sort by positive percentage and take top N
    plot_df = contact_stats.copy()
    plot_df = plot_df.sort_values(pct_col, ascending=False).head(top_n).reset_index(drop=True)
    
    # Create truncated labels for display (first 8 chars + '...' if longer)
    def truncate_label(label: str, max_len: int = 8) -> str:
        if len(str(label)) > max_len:
            return str(label)[:max_len] + '...'
        return str(label)
    
    truncated_labels = [truncate_label(str(c)) for c in plot_df['contact']]
    full_labels = [str(c) for c in plot_df['contact']]
    
    # Use categorical positions for x-axis
    x_positions = list(range(len(plot_df)))
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=x_positions,
            y=plot_df[pct_col],
            marker=dict(
                color=plot_df[pct_col],
                colorscale='Greens',
                showscale=True,
                colorbar=dict(title="% Positive")
            ),
            text=[f"{pct:.1f}%" for pct in plot_df[pct_col]],
            textposition='auto',
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>' +
                'Positive: %{y:.1f}%<br>' +
                f'Total messages: %{{customdata[1]}}<br>' +
                '<extra></extra>'
            ),
            customdata=list(zip(full_labels, plot_df[total_col]))
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Contact",
            yaxis_title="% Positive Sentiment",
            height=400,
            showlegend=False,
            yaxis=dict(range=[0, 100]),
            xaxis={
                'tickmode': 'array',
                'tickvals': x_positions,
                'ticktext': truncated_labels,
                'tickangle': -45
            }
        )
        
        return fig
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(x_positions, plot_df[pct_col], color='#2ecc71', alpha=0.7, width=0.6)
        
        # Add value labels on top of bars
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            ax.text(i, row[pct_col] + 1, f"{row[pct_col]:.1f}%", 
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Contact')
        ax.set_ylabel('% Positive Sentiment')
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(truncated_labels, rotation=-45, ha='left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
