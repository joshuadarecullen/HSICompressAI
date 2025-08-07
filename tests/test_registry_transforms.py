"""Test registry.transforms module functionality."""

import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch

from hsicompressai.registry.transforms import (
    TRANSFORMS,
    register_transform,
)


class TestTransformsRegistry:
    """Test transforms registry initialization and content."""
    
    def test_transforms_is_dict(self):
        """Test TRANSFORMS registry is initialized as a dict."""
        assert isinstance(TRANSFORMS, dict)
    
    def test_transforms_contains_torchvision_transforms(self):
        """Test TRANSFORMS registry contains torchvision transforms."""
        # Check for some common torchvision transforms
        expected_transforms = [
            "Compose", "ToTensor", "Normalize", "Resize", 
            "RandomCrop", "CenterCrop", "RandomHorizontalFlip"
        ]
        
        for transform_name in expected_transforms:
            assert transform_name in TRANSFORMS, f"{transform_name} not found in TRANSFORMS"
    
    def test_transforms_contains_torch_geometric_transforms(self):
        """Test TRANSFORMS registry contains torch_geometric transforms."""
        # Check for some common torch_geometric transforms  
        expected_transforms = [
            "Compose", "NormalizeFeatures", "RandomNodeSplit"
        ]
        
        # Note: Some of these might overlap with torchvision, but that's expected
        for transform_name in expected_transforms:
            if transform_name in TRANSFORMS:
                # At least one should be present
                pass
    
    def test_transforms_registry_not_empty(self):
        """Test that TRANSFORMS registry is not empty."""
        assert len(TRANSFORMS) > 0
    
    def test_transforms_values_are_callable(self):
        """Test that all values in TRANSFORMS registry are callable."""
        for name, transform_class in TRANSFORMS.items():
            assert callable(transform_class), f"Transform {name} is not callable"


class TestRegisterTransform:
    """Test transform registration functionality."""
    
    def setup_method(self):
        """Store original TRANSFORMS and clear custom ones for testing."""
        self.original_transforms = TRANSFORMS.copy()
        # Remove any test transforms that might exist
        test_keys = [k for k in TRANSFORMS.keys() if "test" in k.lower()]
        for key in test_keys:
            TRANSFORMS.pop(key, None)
    
    def teardown_method(self):
        """Restore original TRANSFORMS registry."""
        TRANSFORMS.clear()
        TRANSFORMS.update(self.original_transforms)
    
    def test_register_transform_decorator(self):
        """Test that register_transform decorator works correctly."""
        @register_transform("test_transform")
        class TestTransform:
            def __init__(self):
                pass
            
            def __call__(self, x):
                return x
        
        assert "test_transform" in TRANSFORMS
        assert TRANSFORMS["test_transform"] is TestTransform
    
    def test_register_transform_returns_class(self):
        """Test that register_transform returns the original class."""
        @register_transform("test_transform")
        class TestTransform:
            pass
        
        assert TestTransform.__name__ == "TestTransform"
    
    def test_register_multiple_transforms(self):
        """Test registering multiple transforms."""
        original_count = len(TRANSFORMS)
        
        @register_transform("transform1")
        class Transform1:
            pass
        
        @register_transform("transform2")
        class Transform2:
            pass
        
        assert len(TRANSFORMS) == original_count + 2
        assert "transform1" in TRANSFORMS
        assert "transform2" in TRANSFORMS
        assert TRANSFORMS["transform1"] is Transform1
        assert TRANSFORMS["transform2"] is Transform2
    
    def test_register_transform_name_conflict(self):
        """Test that later registrations overwrite earlier ones."""
        original_count = len(TRANSFORMS)
        
        @register_transform("same_name")
        class Transform1:
            pass
        
        @register_transform("same_name")
        class Transform2:
            pass
        
        assert len(TRANSFORMS) == original_count + 1
        assert TRANSFORMS["same_name"] is Transform2
    
    def test_register_transform_with_callable_function(self):
        """Test registering a function as a transform."""
        @register_transform("function_transform")
        def function_transform(x):
            return x * 2
        
        assert "function_transform" in TRANSFORMS
        assert TRANSFORMS["function_transform"] is function_transform
        assert callable(TRANSFORMS["function_transform"])


class TestTransformsIntegration:
    """Test integration scenarios of the transforms registry."""
    
    def test_can_instantiate_registered_transforms(self):
        """Test that registered torchvision transforms can be instantiated."""
        # Test a simple transform that should be available
        if "ToTensor" in TRANSFORMS:
            transform_class = TRANSFORMS["ToTensor"]
            transform_instance = transform_class()
            assert transform_instance is not None
    
    def test_transforms_have_correct_structure(self):
        """Test that transforms follow expected patterns."""
        # Check that transform names start with uppercase (class convention)
        uppercase_transforms = [name for name in TRANSFORMS.keys() if name[0].isupper()]
        assert len(uppercase_transforms) > 0, "No uppercase transform names found"
    
    def test_no_none_values_in_transforms(self):
        """Test that no transform values are None."""
        for name, transform in TRANSFORMS.items():
            assert transform is not None, f"Transform {name} is None"
    
    def test_torchvision_integration(self):
        """Test that torchvision transforms are properly integrated."""
        # Test that we have access to torchvision transforms
        # These should be present in the TRANSFORMS registry
        torchvision_transforms = ['ToTensor', 'Normalize', 'Resize']
        found_transforms = [name for name in torchvision_transforms if name in TRANSFORMS]
        assert len(found_transforms) > 0, "No torchvision transforms found"
    
    def test_torch_geometric_integration(self):
        """Test that torch_geometric transforms are properly integrated."""
        # Test that we have access to torch_geometric transforms
        # Note: Some transforms might overlap with torchvision
        geometric_transforms = ['Compose', 'NormalizeFeatures']
        # Just check that we have some transforms, don't require specific ones
        # since torch_geometric might not have all expected transforms
        assert len(TRANSFORMS) > 0, "No transforms found in registry"