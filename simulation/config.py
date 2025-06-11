# config.py

# --- Simulation Parameters ---
SIMULATION_CONFIG = {
    "N_CUSTOMERS": 50000,  # Number of customers
    "N_SIMULATION_STEPS": 10,  # Number of time steps to simulate
    # Product ownership regime:
    # 'single': Customer can own only one instance, ownership persists.
    # 'count': Customer can own multiple instances, purchase increments count.
    "PRODUCT_OWNERSHIP_REGIME": "single",  # or 'count'
    # --- Customer Features (X) ---
    # Define the structure and initial distribution of customer features.
    "CUSTOMER_FEATURES_X": {
        "continuous": ["age", "income"],
        "categorical": {
            "customer_type": ["type_A", "type_B", "type_C"],
            "region": ["north", "south", "east", "west"],
        },
        "binary": ["is_prime_member"],
    },
    "CUSTOMER_FEATURES_CATEGORICAL_LEVELS_TO_DROP": {
        "customer_type": "type_A",
        "region": "north",
    },
    # --- Product Hierarchy (B) ---
    # Define the hierarchy as a list of (parent, child) tuples.
    "PRODUCT_HIERARCHY": [
        ("AllProducts", "HotBeverages"),
        ("AllProducts", "ColdBeverages"),
        ("HotBeverages", "Coffee"),
        ("HotBeverages", "Tea"),
        ("ColdBeverages", "IcedTea"),
        ("ColdBeverages", "Water"),
        ("ColdBeverages", "Cola"),
    ],
    # # Rewards for purchasing leaf products
    # "PRODUCT_REWARDS": {
    #     "Coffee": 5.0,
    #     "Tea": 4.0,
    #     "IcedTea": 3.0,
    #     "Water": 2.0,
    #     "Cola": 6.0,
    # },
    "PRODUCT_REWARDS": {
        "Coffee": 5.0,
        "Tea": 5.0,
        "IcedTea": 5.0,
        "Water": 5.0,
        "Cola": 5.0,
    },
    # --- Action Hierarchy (A) ---
    # Define the hierarchy as a list of (parent, child) tuples.
    "ACTION_HIERARCHY": [
        (
            "NoAction",
            "NoAction",
        ),  # Represents the identity action, needs to be in graph
        ("AllActions", "Group_HotBeverage"),
        ("AllActions", "Group_ColdBeverage"),
        ("Group_HotBeverage", "Marketing_Coffee"),
        ("Group_HotBeverage", "Marketing_Tea"),
        ("Group_ColdBeverage", "Marketing_IcedTea"),
        ("Group_ColdBeverage", "Marketing_Water"),
        ("Group_ColdBeverage", "Marketing_Cola"),
    ],
    # # Costs for taking leaf actions
    # "ACTION_COSTS": {
    #     "NoAction": 0.0,  # Cost of not taking an action
    #     "Marketing_Coffee": 0.01,
    #     "Marketing_Tea": 0.01,
    #     "Marketing_IcedTea": 0.01,
    #     "Marketing_Water": 0.01,
    #     "Marketing_Cola": 0.01,
    # },
    "ACTION_COSTS": {
        "NoAction": 0.0,  # Cost of not taking an action
        "Marketing_Coffee": 0.0,
        "Marketing_Tea": 0.0,
        "Marketing_IcedTea": 0.0,
        "Marketing_Water": 0.0,
        "Marketing_Cola": 0.0,
    },
    # # --- Ground Truth Model Weights ---
    # "GROUND_TRUTH_WEIGHTS": {
    #     "Coffee": {
    #         "intercept": -4,
    #         "X": {"customer_type_B": 0.9, "region_east": -0.8},
    #         "A": {
    #             "Marketing_Coffee": 1.20,
    #             "Group_HotBeverage": 0.50,
    #         },
    #         "interaction": {
    #             "Marketing_Coffee_x_region_west": -0.7,
    #             "Group_HotBeverage_x_region_west": 0.11,
    #         },
    #     },
    #     "Tea": {
    #         "intercept": -4,
    #         "X": {"income": 0.40, "region_south": 0.40, "region_west": -0.80},
    #         "A": {"Marketing_Tea": 1.3, "Group_HotBeverage": 0.66},
    #         "interaction": {"Marketing_Tea_x_age": -0.20},
    #     },
    #     "IcedTea": {
    #         "intercept": -5,
    #         "X": {"region_south": -0.6, "is_prime_member": -0.75},
    #         "A": {
    #             "Group_ColdBeverage": 0.12,
    #             "Marketing_Tea": 0.30,
    #             "Marketing_IcedTea": 1.30,
    #         },
    #         "interaction": {
    #             "Group_ColdBeverage_x_is_prime_member": 0.30,
    #             "IcedTea_x_region_west": 0.30,
    #         },
    #     },
    #     "Water": {
    #         "intercept": -6,
    #         "X": {
    #             "is_prime_member": 0.87,
    #             "region_east": -0.60,
    #             "customer_type_C": 0.80,
    #         },
    #         "A": {
    #             "Marketing_Coffee": -0.2,
    #             "Marketing_Tea": -0.3,
    #             "Marketing_Water": 1.4,
    #             "Group_ColdBeverage": 0.20,
    #         },
    #         "interaction": {},
    #     },
    #     "Cola": {
    #         "intercept": -5,
    #         "X": {"customer_type_C": 0.7, "region_east": -0.88, "region_west": -0.50},
    #         "A": {"Group_HotBeverage": -0.97, "Marketing_Cola": 1.5},
    #         "interaction": {
    #             "Marketing_Cola_x_age": -0.26,
    #             "Group_HotBeverage_x_region_west": 0.35,
    #         },
    #     },
    # },
    # --- Ground Truth Model Weights ---
    "GROUND_TRUTH_WEIGHTS": {
        "Coffee": {
            "intercept": -1,
            "X": {"customer_type_B": 0.3, "region_east": 0.4},
            "A": {
                "Marketing_Coffee": 1.00,
                "Group_HotBeverage": 0.50,
            },
            "interaction": {},
        },
        "Tea": {
            "intercept": -1,
            "X": {"customer_type_B": 0.3, "region_east": 0.4},
            "A": {"Marketing_Tea": 1.00, "Group_HotBeverage": 0.50},
            "interaction": {},
        },
        "IcedTea": {
            "intercept": -1,
            "X": {"customer_type_B": 0.3, "region_east": 0.4},
            "A": {
                "Group_ColdBeverage": 0.50,
                "Marketing_IcedTea": 1.00,
            },
            "interaction": {},
        },
        "Water": {
            "intercept": -1,
            "X": {"customer_type_B": 0.3, "region_east": 0.4},
            "A": {
                "Marketing_Water": 1.00,
                "Group_ColdBeverage": 0.50,
            },
            "interaction": {},
        },
        "Cola": {
            "intercept": -1,
            "X": {"customer_type_B": 0.3, "region_east": 0.4},
            "A": {"Group_ColdBeverage": 0.50, "Marketing_Cola": 1.00},
            "interaction": {},
        },
    },
    # --- Initial Population State (B_0) ---
    "INITIAL_PRODUCT_OWNERSHIP_PROBS": {
        "Coffee": 0.1,
        "Tea": 0.05,
        "Water": 0.2,
    },
    # --- Initial Eligibility Ruling---
    # These probabilities describe the probability of being eligible for a marketing campaign
    # In the real world, there would be a connection to X, but we simulate random eligibility here for simplicity
    # In the real world, the eligibility would (hopefully) also be tackled by the IPW model
    "INITIAL_ELIGIBILITY_PROBS": {
        "NoAction": 1.0,
        "Marketing_Coffee": 0.93,
        "Marketing_Tea": 0.95,
        "Marketing_IcedTea": 0.91,
        "Marketing_Water": 0.97,
        "Marketing_Cola": 0.88,
    },
    # not used currently
    "INTERCEPT_COLUMN_NAME": "intercept",
}
