"""
Global in-memory state — usage statistics và uptime tracking.
"""

import time

_start_time: float = time.time()

_stats: dict = {
    "total_requests": 0,
    "active_requests": 0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
}
