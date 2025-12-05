"""
Sentiment Analysis Module (RQ5).

Uses VADER sentiment analysis to analyze the emotional tone of text messages.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Import VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    SentimentIntensityAnalyzer = None


def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sentiment scores for all messages using VADER.
    
    This function operates on RAW, UNTOKENIZED text.
    Results are cached by adding new columns to the dataframe.
    
    Args:
        df: DataFrame with 'text' column containing raw message text
        
    Returns:
        DataFrame with added columns:
        - sentiment_score: VADER compound score (-1 to +1)
        - sentiment_label: 'positive', 'negative', or 'neutral'
    """
    if not VADER_AVAILABLE:
        raise ImportError(
            "vaderSentiment not installed. Install with: pip install vaderSentiment"
        )
    
    # Initialize VADER
    analyzer = SentimentIntensityAnalyzer()
    
    df = df.copy()
    
    # Compute sentiment for each message
    sentiments = []
    labels = []
    
    for text in df['text'].fillna(''):
        if not text or not text.strip():
            # Empty message
            sentiments.append(0.0)
            labels.append('neutral')
        else:
            # Get VADER scores
            scores = analyzer.polarity_scores(text)
            compound = scores['compound']
            
            # Apply standard VADER thresholds
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            sentiments.append(compound)
            labels.append(label)
    
    df['sentiment_score'] = sentiments
    df['sentiment_label'] = labels
    
    return df


def global_sentiment_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute global sentiment statistics.
    
    Args:
        df: DataFrame with sentiment_score and sentiment_label columns
        
    Returns:
        Dictionary with overall and direction-specific sentiment stats
    """
    if 'sentiment_label' not in df.columns or 'sentiment_score' not in df.columns:
        raise ValueError("DataFrame must have sentiment_score and sentiment_label columns. Run compute_sentiment() first.")
    
    summary = {}
    
    # Overall statistics
    total = len(df)
    summary['overall'] = {
        'total_messages': total,
        'avg_sentiment': df['sentiment_score'].mean(),
        'median_sentiment': df['sentiment_score'].median(),
        'pct_positive': (df['sentiment_label'] == 'positive').sum() / total * 100,
        'pct_negative': (df['sentiment_label'] == 'negative').sum() / total * 100,
        'pct_neutral': (df['sentiment_label'] == 'neutral').sum() / total * 100,
    }
    
    # By direction (sent vs received)
    if 'direction' in df.columns:
        for direction in df['direction'].unique():
            dir_df = df[df['direction'] == direction]
            dir_total = len(dir_df)
            
            if dir_total > 0:
                summary[direction] = {
                    'total_messages': dir_total,
                    'avg_sentiment': dir_df['sentiment_score'].mean(),
                    'median_sentiment': dir_df['sentiment_score'].median(),
                    'pct_positive': (dir_df['sentiment_label'] == 'positive').sum() / dir_total * 100,
                    'pct_negative': (dir_df['sentiment_label'] == 'negative').sum() / dir_total * 100,
                    'pct_neutral': (dir_df['sentiment_label'] == 'neutral').sum() / dir_total * 100,
                }
    
    return summary


def sentiment_over_time(df: pd.DataFrame, freq: str = 'M') -> pd.DataFrame:
    """
    Compute sentiment trends over time.
    
    Args:
        df: DataFrame with sentiment scores and timestamp_local
        freq: Pandas frequency string ('M' for month, 'W' for week, 'D' for day)
        
    Returns:
        DataFrame with time-based sentiment aggregates
    """
    if 'sentiment_label' not in df.columns or 'sentiment_score' not in df.columns:
        raise ValueError("DataFrame must have sentiment columns. Run compute_sentiment() first.")
    
    if 'timestamp_local' not in df.columns:
        raise ValueError("DataFrame must have timestamp_local column.")
    
    df = df.copy()
    df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])
    df['period'] = df['timestamp_local'].dt.to_period(freq)
    
    # Group by period
    time_stats = []
    
    for period in df['period'].unique():
        period_df = df[df['period'] == period]
        period_total = len(period_df)
        
        stats = {
            'period': period.to_timestamp(),
            'total_messages': period_total,
            'avg_sentiment': period_df['sentiment_score'].mean(),
            'pct_positive': (period_df['sentiment_label'] == 'positive').sum() / period_total * 100,
            'pct_negative': (period_df['sentiment_label'] == 'negative').sum() / period_total * 100,
            'pct_neutral': (period_df['sentiment_label'] == 'neutral').sum() / period_total * 100,
        }
        
        # Add direction-specific stats if available
        if 'direction' in df.columns:
            for direction in ['in', 'out']:
                dir_df = period_df[period_df['direction'] == direction]
                dir_total = len(dir_df)
                
                if dir_total > 0:
                    stats[f'avg_sentiment_{direction}'] = dir_df['sentiment_score'].mean()
                    stats[f'pct_positive_{direction}'] = (dir_df['sentiment_label'] == 'positive').sum() / dir_total * 100
                    stats[f'pct_negative_{direction}'] = (dir_df['sentiment_label'] == 'negative').sum() / dir_total * 100
                    stats[f'total_{direction}'] = dir_total
                else:
                    stats[f'avg_sentiment_{direction}'] = np.nan
                    stats[f'pct_positive_{direction}'] = np.nan
                    stats[f'pct_negative_{direction}'] = np.nan
                    stats[f'total_{direction}'] = 0
        
        time_stats.append(stats)
    
    result_df = pd.DataFrame(time_stats).sort_values('period')
    return result_df


def per_chat_sentiment(df: pd.DataFrame, chat_identifier: str) -> Dict[str, Any]:
    """
    Compute sentiment statistics for a specific chat.
    
    Args:
        df: DataFrame with sentiment columns
        chat_identifier: Value to match in 'chat_name' or 'participant' column
        
    Returns:
        Dictionary with chat-specific sentiment stats and time series
    """
    if 'sentiment_label' not in df.columns or 'sentiment_score' not in df.columns:
        raise ValueError("DataFrame must have sentiment columns. Run compute_sentiment() first.")
    
    # Filter to specific chat
    if 'chat_name' in df.columns:
        chat_df = df[(df['chat_name'] == chat_identifier) | (df['participant'] == chat_identifier)]
    else:
        chat_df = df[df['participant'] == chat_identifier]
    
    if len(chat_df) == 0:
        return None
    
    # Overall stats for this chat
    total = len(chat_df)
    result = {
        'chat_identifier': chat_identifier,
        'total_messages': total,
        'overall': {
            'avg_sentiment': chat_df['sentiment_score'].mean(),
            'median_sentiment': chat_df['sentiment_score'].median(),
            'pct_positive': (chat_df['sentiment_label'] == 'positive').sum() / total * 100,
            'pct_negative': (chat_df['sentiment_label'] == 'negative').sum() / total * 100,
            'pct_neutral': (chat_df['sentiment_label'] == 'neutral').sum() / total * 100,
        }
    }
    
    # By direction
    if 'direction' in chat_df.columns:
        for direction in ['in', 'out']:
            dir_df = chat_df[chat_df['direction'] == direction]
            dir_total = len(dir_df)
            
            if dir_total > 0:
                result[direction] = {
                    'total_messages': dir_total,
                    'avg_sentiment': dir_df['sentiment_score'].mean(),
                    'median_sentiment': dir_df['sentiment_score'].median(),
                    'pct_positive': (dir_df['sentiment_label'] == 'positive').sum() / dir_total * 100,
                    'pct_negative': (dir_df['sentiment_label'] == 'negative').sum() / dir_total * 100,
                    'pct_neutral': (dir_df['sentiment_label'] == 'neutral').sum() / dir_total * 100,
                }
    
    # Time series for this chat (weekly)
    result['time_series'] = sentiment_over_time(chat_df, freq='W')
    
    return result


def per_contact_sentiment_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sentiment statistics per contact.
    
    Args:
        df: DataFrame with sentiment columns and participant/direction columns
        
    Returns:
        DataFrame with per-contact sentiment statistics
    """
    if 'sentiment_label' not in df.columns or 'sentiment_score' not in df.columns:
        raise ValueError("DataFrame must have sentiment columns. Run compute_sentiment() first.")
    
    if 'participant' not in df.columns:
        raise ValueError("DataFrame must have participant column.")
    
    contact_stats = []
    
    for contact in df['participant'].unique():
        if pd.isna(contact) or contact == "UNKNOWN":
            continue
        
        contact_df = df[df['participant'] == contact]
        total = len(contact_df)
        
        if total == 0:
            continue
        
        # Overall stats
        overall_positive = (contact_df['sentiment_label'] == 'positive').sum()
        overall_negative = (contact_df['sentiment_label'] == 'negative').sum()
        overall_neutral = (contact_df['sentiment_label'] == 'neutral').sum()
        
        stats = {
            'contact': contact,
            'total_messages': total,
            'avg_sentiment': contact_df['sentiment_score'].mean(),
            'pct_positive_overall': (overall_positive / total * 100) if total > 0 else 0,
            'pct_negative_overall': (overall_negative / total * 100) if total > 0 else 0,
            'pct_neutral_overall': (overall_neutral / total * 100) if total > 0 else 0,
        }
        
        # By direction if available
        if 'direction' in df.columns:
            # Sent messages
            sent_df = contact_df[contact_df['direction'] == 'out']
            sent_total = len(sent_df)
            if sent_total > 0:
                sent_positive = (sent_df['sentiment_label'] == 'positive').sum()
                stats['sent_total'] = sent_total
                stats['sent_avg_sentiment'] = sent_df['sentiment_score'].mean()
                stats['pct_positive_sent'] = (sent_positive / sent_total * 100)
                stats['pct_negative_sent'] = ((sent_df['sentiment_label'] == 'negative').sum() / sent_total * 100)
            else:
                stats['sent_total'] = 0
                stats['sent_avg_sentiment'] = np.nan
                stats['pct_positive_sent'] = 0
                stats['pct_negative_sent'] = 0
            
            # Received messages
            recv_df = contact_df[contact_df['direction'] == 'in']
            recv_total = len(recv_df)
            if recv_total > 0:
                recv_positive = (recv_df['sentiment_label'] == 'positive').sum()
                stats['received_total'] = recv_total
                stats['received_avg_sentiment'] = recv_df['sentiment_score'].mean()
                stats['pct_positive_received'] = (recv_positive / recv_total * 100)
                stats['pct_negative_received'] = ((recv_df['sentiment_label'] == 'negative').sum() / recv_total * 100)
            else:
                stats['received_total'] = 0
                stats['received_avg_sentiment'] = np.nan
                stats['pct_positive_received'] = 0
                stats['pct_negative_received'] = 0
        
        contact_stats.append(stats)
    
    result_df = pd.DataFrame(contact_stats)
    return result_df
