#!/usr/bin/env python3
"""
Registry System Demonstration
Shows how the registry system works in HSICompressAI
"""

# Step 1: Import registry functions
from hsicompressai.registry import (
    register_model, 
    register_criterion, 
    MODELS, 
    CRITERIONS
)
import torch
import torch.nn as nn

print("=== HSICompressAI Registry System Demo ===\n")

# Step 2: Register classes using decorators
print("1. Registering classes with decorators:")

@register_model("DemoEncoder")
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        return torch.relu(self.linear(x))

@register_criterion("DemoLoss")
class CustomLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred, target):
        return self.weight * nn.functional.mse_loss(pred, target)

print(f"✓ Registered SimpleEncoder as 'DemoEncoder'")
print(f"✓ Registered CustomLoss as 'DemoLoss'")

# Step 3: Show what's in the registry
print(f"\n2. Registry contents:")
print(f"Available models: {list(MODELS.keys())[-5:]}")  # Show last 5
print(f"Available criterions: {list(CRITERIONS.keys())[-5:]}")  # Show last 5

# Step 4: Dynamic instantiation from registry
print(f"\n3. Dynamic instantiation:")

def create_component_from_config(component_type, name, **kwargs):
    """Factory function that creates components from registry"""
    registries = {
        'model': MODELS,
        'criterion': CRITERIONS
    }
    
    if component_type not in registries:
        raise ValueError(f"Unknown component type: {component_type}")
    
    registry = registries[component_type]
    if name not in registry:
        raise ValueError(f"Component '{name}' not found in {component_type} registry")
    
    component_class = registry[name]
    return component_class(**kwargs)

# Create components by name
encoder = create_component_from_config('model', 'DemoEncoder', input_dim=100, hidden_dim=50)
loss_fn = create_component_from_config('criterion', 'DemoLoss', weight=2.0)

print(f"✓ Created encoder: {encoder.__class__.__name__}")
print(f"✓ Created loss function: {loss_fn.__class__.__name__}")

# Step 5: Show it working
print(f"\n4. Testing the created components:")

# Generate dummy data
x = torch.randn(32, 100)  # batch_size=32, input_dim=100
target = torch.randn(32, 50)  # target with hidden_dim=50

# Forward pass
with torch.no_grad():
    output = encoder(x)
    loss = loss_fn(output, target)

print(f"✓ Input shape: {x.shape}")
print(f"✓ Output shape: {output.shape}")
print(f"✓ Loss value: {loss.item():.4f}")

# Step 6: Configuration-driven example
print(f"\n5. Configuration-driven instantiation:")

# This simulates loading from a config file
config = {
    "model": {
        "name": "DemoEncoder",
        "params": {"input_dim": 784, "hidden_dim": 128}
    },
    "criterion": {
        "name": "DemoLoss", 
        "params": {"weight": 0.5}
    }
}

print(f"Config: {config}")

# Create components from config
model = create_component_from_config('model', config['model']['name'], **config['model']['params'])
criterion = create_component_from_config('criterion', config['criterion']['name'], **config['criterion']['params'])

print(f"✓ Model from config: {model}")
print(f"✓ Criterion from config: {criterion}")

print(f"\n=== Demo Complete ===")
print(f"The registry system enables:")
print(f"• Decoupled registration via decorators")
print(f"• Dynamic component discovery") 
print(f"• Configuration-driven instantiation")
print(f"• Plugin-like extensibility")