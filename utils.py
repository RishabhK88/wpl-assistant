import os
import requests
import json
import re

import streamlit as st

from statistics import mean, stdev
from numpy import percentile

from typing import List, Dict, Tuple

MIN_SCORE_THRESHOLD = 0.1


headers = {
    'Content-Type': 'application/json',
    'Accept': '*/*'
}

def thumbs_feedback(feedback, **kwargs):
    """
    Sends feedback to Amplitude Analytics
    """
    
    send_amplitude_data(
        user_query=kwargs.get("user_query", "No user input"),
        chat_response=kwargs.get("chat_response", "No bot response"),
        demo_name=kwargs.get("demo_name", "Unknown"),
        language = kwargs.get("response_language", "Unknown"),
        feedback=feedback["score"],
    )    
    st.session_state.feedback_key += 1

def send_amplitude_data(user_query, chat_response, demo_name, language, feedback=None):
    amplitude_api_key = os.getenv('AMPLITUDE_TOKEN')
    if not amplitude_api_key:
        return
    data = {
        "api_key": amplitude_api_key,
        "events": [{
            "device_id": st.session_state.device_id,
            "event_type": "submitted_query",
            "event_properties": {
                "Space Name": demo_name,
                "Demo Type": "chatbot",
                "query": user_query,
                "response": chat_response,
                "Response Language": language
            }
        }]
    }
    if feedback:
        data["events"][0]["event_properties"]["feedback"] = feedback

    response = requests.post('https://api2.amplitude.com/2/httpapi', headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        print(f"Amplitude request failed with status code {response.status_code}. Response Text: {response.text}")

def escape_dollars_outside_latex(text):
    pattern = re.compile(r'(\$\$.*?\$\$|\$.*?\$)')
    latex_matches = pattern.findall(text)
    
    placeholders = {}
    for i, match in enumerate(latex_matches):
        placeholder = f'__LATEX_PLACEHOLDER_{i}__'
        placeholders[placeholder] = match
        text = text.replace(match, placeholder)
    
    text = text.replace('$', '\\$')
    
    for placeholder, original in placeholders.items():
        text = text.replace(placeholder, original)
    return text


# region statistical analysis

def _calculate_dynamic_thresholds(scores: List[float], window: List[float]) -> Dict[str, float]:
    """Calculate dynamic thresholds based on data characteristics"""
    window_mean = mean(window)
    window_std = stdev(window) if len(window) > 1 else 0
    
    # Calculate coefficient of variation to measure score dispersion
    cv = window_std / window_mean if window_mean > 0 else 0
    
    # Calculate score distribution characteristics
    q75, q25 = percentile(scores, [75, 25])
    iqr = q75 - q25
    
    return {
        'local_drop_threshold': max(
            30,  # minimum threshold
            min(75,  # maximum threshold
                50 * (1 + cv)  # adjust based on variation
            )
        ),
        'baseline_drop_threshold': max(
            25,  # minimum threshold
            min(60,  # maximum threshold
                40 * (1 + cv)  # adjust based on variation
            )
        ),
        'std_multiplier': max(
            0.5,  # minimum multiplier
            min(2.0,  # maximum multiplier
                1 + (iqr / (q75 + 0.001))  # adjust based on spread
            )
        )
    }


def find_first_relevance_drop(search_results: List[Dict], 
                            window_size: int = 3,
                            min_scores_needed: int = 4) -> Tuple[int, Dict]:
    """
    Find the first significant drop in relevance scores using sliding window analysis
    with dynamic thresholds based on dataset characteristics.
    
    Args:
        search_results: List of search result dictionaries
        window_size: Size of sliding window for local trend analysis
        min_scores_needed: Minimum number of scores needed for analysis
        
    Returns:
        Tuple of (cutoff_index, analysis_metrics)
    """
    if not search_results or len(search_results) < min_scores_needed:
        return len(search_results) if search_results else 0, {}
    
    scores = [result['score'] for result in search_results]
    
    metrics = {}
    
    # Need at least window_size + 1 scores to detect a drop
    if len(scores) < window_size + 1:
        return len(scores), metrics
    
    # Analyze first window to establish baseline
    baseline_window = scores[:window_size]
    baseline_mean = mean(baseline_window)
    baseline_std = stdev(baseline_window) if len(baseline_window) > 1 else 0
    
    thresholds = _calculate_dynamic_thresholds(scores, baseline_window)
    metrics['thresholds'] = thresholds
    
    # Function to calculate relative drop
    def calculate_drop(prev_value: float, curr_value: float) -> float:
        return ((prev_value - curr_value) / prev_value) * 100 if prev_value > 0 else 0
    
    # Look for first significant drop
    for i in range(window_size, len(scores)):
        prev_score = scores[i-1]
        curr_score = scores[i]
       
        # Calculate local and global statistics
        local_drop = calculate_drop(prev_score, curr_score)
        relative_to_baseline = calculate_drop(baseline_mean, curr_score)
        
        # Record drop information
        drop_info = {
            'index': i,
            'prev_score': prev_score,
            'curr_score': curr_score,
            'local_drop_percent': local_drop,
            'relative_to_baseline_percent': relative_to_baseline
        }
        
        is_significant_drop = (
            # Local drop criterion: significant drop from previous score
            local_drop > thresholds['local_drop_threshold'] and
            # Baseline criterion: significant drop from initial baseline
            relative_to_baseline > thresholds['baseline_drop_threshold'] and
            # Absolute value criterion: score is notably lower
            curr_score < (baseline_mean - (thresholds['std_multiplier'] * baseline_std))
        )
        
        if is_significant_drop:
            metrics['cutoff_type'] = 'significant_drop'
            metrics['cutoff_details'] = drop_info
            return metrics
        elif curr_score < MIN_SCORE_THRESHOLD:
            metrics['cutoff_type'] = 'min_threshold_reached'
            metrics['cutoff_details'] = drop_info
            return metrics

    
    metrics['cutoff_type'] = 'no_significant_drop'
    return metrics

# endregion statistical analysis
