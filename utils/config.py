"""
Configuration: weather states, transitions, rewards, and outfits.
"""

# --- Weather model ---

weather_states = ["Cold", "Mild", "Hot", "Rainy"]

# Transition probabilities between weather states.
# Rows sum to 1.0. Order corresponds to weather_states.
weather_transition_matrix = [
    # To: Cold  Mild  Hot   Rainy
    [0.6, 0.2, 0.1, 0.1],  # From Cold
    [0.2, 0.5, 0.2, 0.1],  # From Mild
    [0.1, 0.3, 0.5, 0.1],  # From Hot
    [0.2, 0.2, 0.1, 0.5],  # From Rainy
]

# --- Simple category-based MDP config ---

outfit_categories = ["Warm", "Light", "Casual", "Formal", "Rain-Ready"]

reward_weights_simple = {
    "comfort": 5.0,
    "mismatch_penalty": 6.0,
    "variety": 2.0,
    "repeat_penalty": 3.0,
}

# --- Finite-wardrobe MDP config ---

# Example wardrobe: you can customize these to be more fun / realistic.
outfits = [
    {
        "name": "Pink Raincoat Fit",
        "category": "Rain-Ready",
        "rain_ready": True,
        "warmth": "warm",
    },
    {
        "name": "Cozy Sweater & Jeans",
        "category": "Warm",
        "rain_ready": False,
        "warmth": "warm",
    },
    {
        "name": "Sundress",
        "category": "Light",
        "rain_ready": False,
        "warmth": "light",
    },
    {
        "name": "Casual Tee & Shorts",
        "category": "Casual",
        "rain_ready": False,
        "warmth": "light",
    },
    {
        "name": "Business Dress",
        "category": "Formal",
        "rain_ready": False,
        "warmth": "mild",
    },
    {
        "name": "Hoodie & Leggings",
        "category": "Casual",
        "rain_ready": False,
        "warmth": "warm",
    },
    {
        "name": "Black Blazer Set",
        "category": "Formal",
        "rain_ready": False,
        "warmth": "mild",
    },
]

reward_weights_wardrobe = {
    "comfort": 5.0,
    "mismatch_penalty": 6.0,
}
