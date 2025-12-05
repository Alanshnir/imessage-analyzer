"""
GPT-Powered Texting Behavior Chatbot (RQ6).

This module provides a chatbot interface that answers questions about texting patterns
using ONLY aggregated statistics from RQ1-5.

PRIVACY: NO raw message texts, phone numbers, or personally identifiable information
are sent to the OpenAI API. Only aggregated statistics and patterns are included.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import json

# Import OpenAI at module level so we can see real errors
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


def build_behavior_summary(rq_results: Dict[str, Any], deidentify: bool = True) -> str:
    """
    Build a comprehensive behavior summary from RQ1-5 results.
    
    PRIVACY: This function ONLY includes aggregated statistics.
    NO raw messages, phone numbers, emails, or PII are included.
    
    Args:
        rq_results: Dictionary containing results from all research questions
        deidentify: If True, anonymize contacts. If False, use actual contact names.
        
    Returns:
        Text summary of aggregated texting behavior patterns
    """
    summary_parts = []
    
    summary_parts.append("# User's iMessage Behavior Summary (Aggregated Statistics Only)")
    summary_parts.append("\nThis summary contains ONLY aggregated patterns and statistics.")
    summary_parts.append("NO raw messages or personally identifiable information is included.")
    summary_parts.append("With fewer RQ sections, we provide MORE DETAIL per topic to give you richer insights.")
    summary_parts.append("\n**IMPORTANT**: RQ1, RQ2, and RQ4 all use the SAME topics learned from all aggregated messages (sent + received).")
    summary_parts.append("The topics are identical across these RQs - they are just analyzed from different perspectives.\n")
    
    # RQ1: Topics User Tends Not to Engage With
    if 'rq1_df' in rq_results and rq_results['rq1_df'] is not None:
        rq1_df = rq_results['rq1_df']
        num_rq1_topics = len(rq1_df)
        summary_parts.append(f"\n## RQ1: Topics User Tends Not to Engage With ({num_rq1_topics} Topics)")
        summary_parts.append("**Uses topics learned from ALL aggregated messages (sent + received).**")
        summary_parts.append("These are topics where the user shows high reluctance (slow or no replies) on RECEIVED messages.")
        
        if num_rq1_topics > 0:
            summary_parts.append(f"- Total topics analyzed: {num_rq1_topics}")
            summary_parts.append("- ALL topics with full details (ranked by reluctance):")
            
            for _, row in rq1_df.iterrows():
                # Now we can show MORE words (up to 15) since we removed RQ5-7
                words_full = ', '.join(row['top_words'].split(', ')[:15])
                summary_parts.append(
                    f"  * Topic {row['topic_id']}: {words_full}"
                )
                summary_parts.append(
                    f"    Reluctance: {row['reluctance_score']:.3f} | "
                    f"Frequency: {row['frequency']} messages | "
                    f"Final Rank: {row['final_rank']:.3f}"
                )
    
    # RQ2: Most Commonly Discussed Topics
    if 'rq2_topic_words' in rq_results and rq_results['rq2_topic_words'] is not None:
        rq2_df = rq_results['rq2_topic_words']
        num_rq2_topics = len(rq2_df)
        summary_parts.append(f"\n## RQ2: Most Commonly Discussed Topics ({num_rq2_topics} Topics)")
        summary_parts.append("**Uses the SAME topics from RQ1 (learned from all aggregated messages).**")
        summary_parts.append("Topics ranked by how frequently they appear in ALL conversations (sent + received).")
        
        if num_rq2_topics > 0:
            # Sort by message count
            rq2_sorted = rq2_df.sort_values('message_count', ascending=False)
            summary_parts.append(f"- Total topics: {num_rq2_topics} (same topics as RQ1, different analysis)")
            summary_parts.append("- ALL topics with full details (ranked by frequency across all messages):")
            
            for _, row in rq2_sorted.iterrows():
                # Show MORE words (up to 15) for richer context
                words_full = ', '.join(row['top_words'].split(', ')[:15])
                summary_parts.append(
                    f"  * Topic {row['topic_id']}: {words_full}"
                )
                summary_parts.append(
                    f"    Message count (all messages): {row['message_count']}"
                )
    
    # RQ2 Per-Contact Topic Mix (top 10 contacts only for efficiency)
    if 'rq2_contact_topic' in rq_results and rq_results['rq2_contact_topic'] is not None:
        contact_topic_df = rq_results['rq2_contact_topic']
        if len(contact_topic_df) > 0:
            if deidentify:
                summary_parts.append("\n- Per-contact topic preferences (top 10 contacts, anonymized):")
            else:
                summary_parts.append("\n- Per-contact topic preferences (top 10 contacts):")
            for idx, row in contact_topic_df.head(10).iterrows():
                # Get top 2 topics for this contact
                topic_cols = [col for col in row.index if str(col).startswith('topic_')]
                if topic_cols:
                    top_topics = sorted([(col, row[col]) for col in topic_cols], 
                                       key=lambda x: x[1], reverse=True)[:2]
                    if deidentify:
                        contact_label = f"Contact_{chr(65 + idx % 26)}"
                    else:
                        # Use actual contact name - prefer participant column
                        contact_label = str(row.get('participant', row.get('contact', f"Contact_{idx}")))
                        # Clean up the label if it's a phone number or email
                        if contact_label and contact_label not in ['Unknown', 'UNKNOWN']:
                            contact_label = contact_label
                        else:
                            contact_label = f"Contact_{idx}"
                    topic_desc = ', '.join([f"T{col.split('_')[1]}:{val:.0%}" 
                                           for col, val in top_topics])
                    summary_parts.append(f"  * {contact_label}: {topic_desc}")
    
    # RQ3: Group vs One-to-One Responsiveness
    if 'rq3_stats' in rq_results and rq_results['rq3_stats'] is not None:
        rq3_stats = rq_results['rq3_stats']
        summary_parts.append("\n## RQ3: Group vs One-to-One Responsiveness")
        
        if 'by_category' in rq3_stats:
            by_cat = rq3_stats['by_category']
            summary_parts.append("- Reply rates by conversation type:")
            for _, row in by_cat.iterrows():
                summary_parts.append(
                    f"  * {row['category']}: {row['reply_rate']:.1%} reply rate, "
                    f"median response: {row['median_response_time']:.1f} min, "
                    f"mean response: {row['mean_response_time']:.1f} min, "
                    f"{row['count']} messages"
                )
        
        if 'contact_comparison' in rq3_stats:
            contact_comp = rq3_stats['contact_comparison'].copy()
            contact_comp['reply_diff'] = contact_comp['group_reply_rate'] - contact_comp['one_on_one_reply_rate']
            contact_comp['total_msgs'] = contact_comp['one_on_one_count'] + contact_comp['group_count']
            
            # Sort by reply difference
            contact_comp_sorted = contact_comp.sort_values('reply_diff', ascending=False)
            
            summary_parts.append(f"- Contact-level reply behavior (total: {len(contact_comp_sorted)} contacts):")
            if deidentify:
                summary_parts.append("  Contacts anonymized as Contact_A, Contact_B, etc.")
            
            # Limit to top 20 contacts for API efficiency
            for idx, row in contact_comp_sorted.head(20).iterrows():
                if deidentify:
                    contact_label = f"Contact_{chr(65 + idx % 26)}"
                else:
                    # Use actual contact name - prefer participant column
                    contact_label = str(row.get('participant', f"Contact_{idx}"))
                    # Clean up the label
                    if contact_label in ['Unknown', 'UNKNOWN', '']:
                        contact_label = f"Contact_{idx}"
                summary_parts.append(
                    f"  * {contact_label}: 1on1={row['one_on_one_reply_rate']:.0%}, "
                    f"Grp={row['group_reply_rate']:.0%}, Diff={row['reply_diff']:+.0%}"
                )
    
    # RQ4: Conversation Starter Topics
    if 'rq4_starters' in rq_results and rq_results['rq4_starters'] is not None:
        rq4_df = rq_results['rq4_starters']
        num_rq4_topics = len(rq4_df)
        summary_parts.append(f"\n## RQ4: Conversation Starter Topics ({num_rq4_topics} Topics)")
        summary_parts.append("**Uses the SAME topics from RQ1 (learned from all aggregated messages).**")
        summary_parts.append("Topics that tend to start conversations (high reply rate, fast response, early in sessions).")
        summary_parts.append("Analyzed on RECEIVED messages to identify which topics prompt quick replies.")
        
        if num_rq4_topics > 0:
            summary_parts.append(f"- Total starter topics: {num_rq4_topics} (same topics as RQ1, different analysis)")
            summary_parts.append("- ALL starter topics with full details (ranked by starter score):")
            
            for _, row in rq4_df.iterrows():
                # Show MORE words (up to 15) for better understanding
                words_full = ', '.join(row['top_words'].split(', ')[:15])
                summary_parts.append(
                    f"  * Topic {row['topic_id']}: {words_full}"
                )
                summary_parts.append(
                    f"    Starter Score: {row['starter_score']:.3f} | "
                    f"Reply Rate: {row['reply_rate']:.1%} | "
                    f"Avg Response: {row['avg_response_time']:.1f}min | "
                    f"Starter Probability: {row['starter_probability']:.1%} | "
                    f"Messages: {row['message_count']}"
                )
    
    # RQ5: Sentiment Analysis
    if 'rq5_sentiment_summary' in rq_results and rq_results['rq5_sentiment_summary'] is not None:
        sentiment_summary = rq_results['rq5_sentiment_summary']
        summary_parts.append("\n## RQ5: Sentiment Analysis of Text Messages")
        summary_parts.append("VADER sentiment analysis measuring emotional tone of messages.")
        summary_parts.append("Sentiment scores range from -1 (very negative) to +1 (very positive).")
        
        if 'overall' in sentiment_summary:
            overall = sentiment_summary['overall']
            summary_parts.append(f"\n- Overall Sentiment (all {overall['total_messages']:,} messages):")
            summary_parts.append(f"  * Average sentiment: {overall['avg_sentiment']:.3f}")
            summary_parts.append(f"  * Positive: {overall['pct_positive']:.1f}%")
            summary_parts.append(f"  * Negative: {overall['pct_negative']:.1f}%")
            summary_parts.append(f"  * Neutral: {overall['pct_neutral']:.1f}%")
        
        if 'out' in sentiment_summary:
            sent = sentiment_summary['out']
            summary_parts.append(f"\n- Sent Messages ({sent['total_messages']:,} messages):")
            summary_parts.append(f"  * Average sentiment: {sent['avg_sentiment']:.3f}")
            summary_parts.append(f"  * Positive: {sent['pct_positive']:.1f}%")
            summary_parts.append(f"  * Negative: {sent['pct_negative']:.1f}%")
            summary_parts.append(f"  * Neutral: {sent['pct_neutral']:.1f}%")
        
        if 'in' in sentiment_summary:
            recv = sentiment_summary['in']
            summary_parts.append(f"\n- Received Messages ({recv['total_messages']:,} messages):")
            summary_parts.append(f"  * Average sentiment: {recv['avg_sentiment']:.3f}")
            summary_parts.append(f"  * Positive: {recv['pct_positive']:.1f}%")
            summary_parts.append(f"  * Negative: {recv['pct_negative']:.1f}%")
            summary_parts.append(f"  * Neutral: {recv['pct_neutral']:.1f}%")
        
        # Include sentiment trend info if available
        if 'rq5_sentiment_trend' in rq_results and rq_results['rq5_sentiment_trend'] is not None:
            trend_df = rq_results['rq5_sentiment_trend']
            if len(trend_df) > 0:
                summary_parts.append("\n- Sentiment Trends Over Time:")
                summary_parts.append(f"  * Analyzed across {len(trend_df)} time periods")
                summary_parts.append(f"  * Recent avg sentiment: {trend_df.iloc[-1]['avg_sentiment']:.3f}")
                summary_parts.append(f"  * Recent positive rate: {trend_df.iloc[-1]['pct_positive']:.1f}%")
    
    # Add overall statistics
    if 'stats' in rq_results:
        stats = rq_results['stats']
        summary_parts.append("\n## Overall Statistics")
        summary_parts.append(f"- Total messages analyzed: {stats.get('total_messages', 'N/A')}")
        summary_parts.append(f"- Messages received: {stats.get('messages_received', 'N/A')}")
        summary_parts.append(f"- Messages sent: {stats.get('messages_sent', 'N/A')}")
        summary_parts.append(f"- Overall reply rate: {stats.get('reply_rate', 0):.1%}")
        summary_parts.append(f"- Median response time: {stats.get('median_response_time', 'N/A'):.1f} minutes")
    
    summary_parts.append("\n---")
    summary_parts.append("\nREMINDER: Base your answers ONLY on these aggregated statistics.")
    summary_parts.append("You do NOT have access to raw messages, contact names, or any PII.")
    
    return '\n'.join(summary_parts)


def run_rq9_chatbot(
    user_question: str,
    behavior_summary: str,
    api_key: str,
    conversation_history: list = None
) -> tuple[str, list]:
    """
    Run the GPT-powered chatbot to answer questions about texting behavior.
    
    PRIVACY: This function only sends aggregated statistics to the OpenAI API.
    NO raw messages, phone numbers, or personally identifiable information is sent.
    
    Args:
        user_question: The user's question
        behavior_summary: Aggregated behavior summary from build_behavior_summary()
        api_key: OpenAI API key
        conversation_history: Previous conversation messages (optional)
        
    Returns:
        Tuple of (assistant response, updated conversation history)
    """
    if not OPENAI_AVAILABLE or OpenAI is None:
        raise ImportError(
            "OpenAI package not installed. Install with: pip install openai"
        )
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Build conversation messages
    messages = []
    
    # System prompt with behavior summary
    system_prompt = f"""You are a helpful assistant that interprets a user's iMessage texting behavior.

You have access ONLY to aggregated statistics and patterns from their texting analysis.
You do NOT have access to raw messages, contact names, phone numbers, or any personally identifiable information.

Base ALL your answers strictly on the aggregated patterns provided below.

If the user asks about specific messages or contacts, explain that you only have access to aggregated patterns, not individual messages.

AGGREGATED BEHAVIOR PATTERNS:
{behavior_summary}

When answering:
- Be insightful and analytical
- Use the specific metrics and patterns provided
- Avoid making up information not in the summary
- If a pattern isn't in the data, say so
- Be friendly and conversational"""

    messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user question
    messages.append({"role": "user", "content": user_question})
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",  # Using gpt-4o (latest, more cost-effective than gpt-4-turbo)
        messages=messages,
        temperature=0.7,
        max_tokens=2500  # Increased since we now have more context space (removed RQ5-7)
    )
    
    # Extract assistant response
    assistant_message = response.choices[0].message.content
    
    # Update conversation history
    updated_history = conversation_history.copy() if conversation_history else []
    updated_history.append({"role": "user", "content": user_question})
    updated_history.append({"role": "assistant", "content": assistant_message})
    
    # Keep only last 10 messages to avoid context overflow
    if len(updated_history) > 20:  # 10 exchanges = 20 messages
        updated_history = updated_history[-20:]
    
    return assistant_message, updated_history


def get_example_questions() -> list:
    """Get example questions users can ask the chatbot."""
    return [
        "What topics do I reply to the fastest?",
        "Why do I avoid certain topics?",
        "How do I text differently in groups vs one-on-one?",
        "Which topics do I usually start conversations with?",
        "Am I more positive or negative in my texts?",
        "How does my sentiment differ in sent vs received messages?",
        "Is there a difference in sentiment between group and one-on-one chats?",
        "What patterns show I'm reluctant to engage?",
        "Compare my sentiment and reply behavior",
        "What topics get no response from me?"
    ]

