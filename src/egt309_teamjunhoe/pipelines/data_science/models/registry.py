# HARISH
# Initializes an empty dictionary to store key-value pairs of names and classes 
MODEL_REGISTRY: dict[str, type] = {}

# Function that accepts a name as an argument (becomes the key)
def register_model(name: str):
    # Returns a function that accepts a class as an argument (becomes the value)
    def decorator(cls):
        # Adds to dictionary
        MODEL_REGISTRY[name] = cls
        # Returns the class so when this is used as a decorator around the class
        # the class will still work fine
        return cls
    return decorator