import random

def calculate_retry_delay(
    attempt: int,
    base_delay: float,
    max_delay: float
) -> float:
    delay = min(
        base_delay * (2 ** attempt),
        max_delay,
    )
    
    jitter = random.uniform(0, delay * 0.25)
    
    return delay + jitter