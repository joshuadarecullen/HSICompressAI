"""Test registry.__init__ module functionality."""

import pytest
from unittest.mock import patch

import hsicompressai.registry as registry


class TestRegistryImports:
    """Test that registry module imports work correctly."""
    
    def test_imports_from_torch(self):
        """Test that imports from registry.torch work correctly."""
        # Test that we can access registries
        assert hasattr(registry, 'CRITERIONS')
        assert hasattr(registry, 'DATASETS')
        assert hasattr(registry, 'MODELS')
        assert hasattr(registry, 'MODULES')
        assert hasattr(registry, 'OPTIMIZERS')
        assert hasattr(registry, 'PLMODULES')
        assert hasattr(registry, 'PLDATAMODULES')
        assert hasattr(registry, 'PLCALLBACKS')
        assert hasattr(registry, 'SCHEDULERS')
        
        # Test that we can access registration functions
        assert hasattr(registry, 'register_criterion')
        assert hasattr(registry, 'register_dataset')
        assert hasattr(registry, 'register_model')
        assert hasattr(registry, 'register_module')
        assert hasattr(registry, 'register_optimizer')
        assert hasattr(registry, 'register_plcallback')
        assert hasattr(registry, 'register_pldatamodule')
        assert hasattr(registry, 'register_plmodule')
        assert hasattr(registry, 'register_scheduler')
    
    def test_imports_from_transforms(self):
        """Test that imports from registry.transforms work correctly."""
        assert hasattr(registry, 'TRANSFORMS')
        assert hasattr(registry, 'register_transform')
    
    def test_all_attribute_exists(self):
        """Test that __all__ attribute exists and contains expected items."""
        assert hasattr(registry, '__all__')
        assert isinstance(registry.__all__, list)
        assert len(registry.__all__) > 0
    
    def test_all_contains_registries(self):
        """Test that __all__ contains all registry dictionaries."""
        expected_registries = [
            "CRITERIONS", "DATASETS", "MODELS", "MODULES", "OPTIMIZERS",
            "PLMODULES", "PLDATAMODULES", "PLCALLBACKS", "SCHEDULERS", "TRANSFORMS"
        ]
        
        for registry_name in expected_registries:
            assert registry_name in registry.__all__, f"{registry_name} not in __all__"
    
    def test_all_contains_register_functions(self):
        """Test that __all__ contains all registration functions."""
        expected_functions = [
            "register_criterion", "register_dataset", "register_model", 
            "register_module", "register_optimizer", "register_plcallback",
            "register_pldatamodule", "register_plmodule", "register_scheduler",
            "register_transform"
        ]
        
        for func_name in expected_functions:
            assert func_name in registry.__all__, f"{func_name} not in __all__"


class TestRegistryTypes:
    """Test that registry objects have correct types."""
    
    def test_registries_are_dicts(self):
        """Test that all registry objects are dictionaries."""
        registries = [
            registry.CRITERIONS, registry.DATASETS, registry.MODELS,
            registry.MODULES, registry.OPTIMIZERS, registry.PLMODULES,
            registry.PLDATAMODULES, registry.PLCALLBACKS, registry.SCHEDULERS,
            registry.TRANSFORMS
        ]
        
        for reg in registries:
            assert isinstance(reg, dict)
    
    def test_register_functions_are_callable(self):
        """Test that all registration functions are callable."""
        register_functions = [
            registry.register_criterion, registry.register_dataset,
            registry.register_model, registry.register_module,
            registry.register_optimizer, registry.register_plcallback,
            registry.register_pldatamodule, registry.register_plmodule,
            registry.register_scheduler, registry.register_transform
        ]
        
        for func in register_functions:
            assert callable(func)


class TestRegistryIntegration:
    """Test integration between different registry modules."""
    
    def test_torch_registries_accessible_through_init(self):
        """Test that torch registries are accessible through __init__."""
        from hsicompressai.registry.torch import MODELS as TORCH_MODELS
        
        # Should be the same object
        assert registry.MODELS is TORCH_MODELS
    
    def test_transforms_registry_accessible_through_init(self):
        """Test that transforms registry is accessible through __init__."""
        from hsicompressai.registry.transforms import TRANSFORMS as TRANS_TRANSFORMS
        
        # Should be the same object  
        assert registry.TRANSFORMS is TRANS_TRANSFORMS
    
    def test_registration_functions_work_through_init(self):
        """Test that registration functions work when accessed through __init__."""
        # Clear any existing test entries
        if "test_model_init" in registry.MODELS:
            del registry.MODELS["test_model_init"]
        
        @registry.register_model("test_model_init")
        class TestModel:
            pass
        
        assert "test_model_init" in registry.MODELS
        assert registry.MODELS["test_model_init"] is TestModel
        
        # Clean up
        del registry.MODELS["test_model_init"]
    
    def test_transform_registration_works_through_init(self):
        """Test that transform registration works when accessed through __init__."""
        original_count = len(registry.TRANSFORMS)
        
        @registry.register_transform("test_transform_init")
        class TestTransform:
            pass
        
        assert "test_transform_init" in registry.TRANSFORMS
        assert registry.TRANSFORMS["test_transform_init"] is TestTransform
        assert len(registry.TRANSFORMS) == original_count + 1
        
        # Clean up
        del registry.TRANSFORMS["test_transform_init"]


class TestRegistryUsagePatterns:
    """Test common usage patterns of the registry system."""
    
    def test_can_import_specific_items(self):
        """Test that specific items can be imported."""
        from hsicompressai.registry import MODELS, register_model
        
        assert MODELS is registry.MODELS
        assert register_model is registry.register_model
    
    def test_can_import_all_from_registry(self):
        """Test that 'from registry import *' pattern works."""
        # This is tested implicitly by the __all__ tests above
        # but we verify the pattern works
        import hsicompressai.registry as reg
        
        for item_name in reg.__all__:
            assert hasattr(reg, item_name)
    
    def test_registry_module_structure(self):
        """Test that the registry module has expected structure."""
        import hsicompressai.registry
        
        # Should have submodules
        assert hasattr(hsicompressai.registry, 'torch')
        assert hasattr(hsicompressai.registry, 'transforms')
        
        # Should have the main interface items
        expected_items = [
            'MODELS', 'TRANSFORMS', 'register_model', 'register_transform'
        ]
        
        for item in expected_items:
            assert hasattr(hsicompressai.registry, item)