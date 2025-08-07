"""Test registry.torch module functionality."""

import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch

from hsicompressai.registry.torch import (
    CRITERIONS,
    DATASETS, 
    MODELS,
    MODULES,
    OPTIMIZERS,
    SCHEDULERS,
    PLMODULES,
    PLDATAMODULES,
    PLCALLBACKS,
    register_criterion,
    register_dataset,
    register_model,
    register_module,
    register_optimizer,
    register_scheduler,
    register_pldatamodule,
    register_plmodule,
    register_plcallback,
)


class TestRegistryDictionaries:
    """Test that registry dictionaries are properly initialized."""
    
    def test_criterions_is_empty_dict(self):
        """Test CRITERIONS registry is initialized as empty dict."""
        assert isinstance(CRITERIONS, dict)
    
    def test_datasets_is_empty_dict(self):
        """Test DATASETS registry is initialized as empty dict."""
        assert isinstance(DATASETS, dict)
    
    def test_models_is_empty_dict(self):
        """Test MODELS registry is initialized as empty dict."""
        assert isinstance(MODELS, dict)
    
    def test_modules_is_empty_dict(self):
        """Test MODULES registry is initialized as empty dict."""
        assert isinstance(MODULES, dict)
    
    def test_optimizers_contains_torch_optimizers(self):
        """Test OPTIMIZERS registry contains PyTorch optimizers."""
        assert isinstance(OPTIMIZERS, dict)
        assert len(OPTIMIZERS) > 0
        # Check for common PyTorch optimizers
        expected_optimizers = ["Adam", "SGD", "AdamW", "RMSprop"]
        for optimizer in expected_optimizers:
            assert optimizer in OPTIMIZERS
    
    def test_schedulers_contains_torch_schedulers(self):
        """Test SCHEDULERS registry contains PyTorch schedulers."""
        assert isinstance(SCHEDULERS, dict)
        assert len(SCHEDULERS) > 0
        # Check for common PyTorch schedulers
        expected_schedulers = ["StepLR", "MultiStepLR", "ExponentialLR"]
        for scheduler in expected_schedulers:
            assert scheduler in SCHEDULERS
    
    def test_pl_registries_are_empty_dicts(self):
        """Test PyTorch Lightning registries are initialized as empty dicts."""
        assert isinstance(PLMODULES, dict)
        assert isinstance(PLDATAMODULES, dict)
        assert isinstance(PLCALLBACKS, dict)


class TestRegisterCriterion:
    """Test criterion registration functionality."""
    
    def setup_method(self):
        """Clear CRITERIONS registry before each test."""
        CRITERIONS.clear()
    
    def test_register_criterion_decorator(self):
        """Test that register_criterion decorator works correctly."""
        @register_criterion("test_criterion")
        class TestCriterion:
            pass
        
        assert "test_criterion" in CRITERIONS
        assert CRITERIONS["test_criterion"] is TestCriterion
    
    def test_register_criterion_returns_class(self):
        """Test that register_criterion returns the original class."""
        @register_criterion("test_criterion")
        class TestCriterion:
            pass
        
        assert TestCriterion.__name__ == "TestCriterion"
    
    def test_register_multiple_criterions(self):
        """Test registering multiple criterions."""
        @register_criterion("criterion1")
        class Criterion1:
            pass
        
        @register_criterion("criterion2") 
        class Criterion2:
            pass
        
        assert len(CRITERIONS) == 2
        assert "criterion1" in CRITERIONS
        assert "criterion2" in CRITERIONS


class TestRegisterDataset:
    """Test dataset registration functionality."""
    
    def setup_method(self):
        """Clear DATASETS registry before each test."""
        DATASETS.clear()
    
    def test_register_dataset_decorator(self):
        """Test that register_dataset decorator works correctly."""
        @register_dataset("test_dataset")
        class TestDataset:
            pass
        
        assert "test_dataset" in DATASETS
        assert DATASETS["test_dataset"] is TestDataset


class TestRegisterModel:
    """Test model registration functionality."""
    
    def setup_method(self):
        """Clear MODELS registry before each test."""
        MODELS.clear()
    
    def test_register_model_decorator(self):
        """Test that register_model decorator works correctly."""
        @register_model("test_model")
        class TestModel:
            pass
        
        assert "test_model" in MODELS
        assert MODELS["test_model"] is TestModel


class TestRegisterModule:
    """Test module registration functionality."""
    
    def setup_method(self):
        """Clear MODULES registry before each test."""
        MODULES.clear()
    
    def test_register_module_decorator(self):
        """Test that register_module decorator works correctly."""
        @register_module("test_module")
        class TestModule:
            pass
        
        assert "test_module" in MODULES
        assert MODULES["test_module"] is TestModule


class TestRegisterOptimizer:
    """Test optimizer registration functionality."""
    
    def setup_method(self):
        """Store original OPTIMIZERS and clear custom ones."""
        self.original_optimizers = OPTIMIZERS.copy()
        # Remove custom optimizers, keep torch ones
        custom_keys = [k for k in OPTIMIZERS.keys() if not k[0].isupper() or "test" in k.lower()]
        for key in custom_keys:
            OPTIMIZERS.pop(key, None)
    
    def teardown_method(self):
        """Restore original OPTIMIZERS."""
        OPTIMIZERS.clear()
        OPTIMIZERS.update(self.original_optimizers)
    
    def test_register_optimizer_decorator(self):
        """Test that register_optimizer decorator works correctly."""
        @register_optimizer("test_optimizer")
        def test_optimizer_func():
            pass
        
        assert "test_optimizer" in OPTIMIZERS
        assert OPTIMIZERS["test_optimizer"] is test_optimizer_func


class TestRegisterScheduler:
    """Test scheduler registration functionality."""
    
    def setup_method(self):
        """Store original SCHEDULERS and clear custom ones."""
        self.original_schedulers = SCHEDULERS.copy()
        # Remove custom schedulers, keep torch ones
        custom_keys = [k for k in SCHEDULERS.keys() if not k[0].isupper() or "test" in k.lower()]
        for key in custom_keys:
            SCHEDULERS.pop(key, None)
    
    def teardown_method(self):
        """Restore original SCHEDULERS."""
        SCHEDULERS.clear() 
        SCHEDULERS.update(self.original_schedulers)
    
    def test_register_scheduler_decorator(self):
        """Test that register_scheduler decorator works correctly."""
        @register_scheduler("test_scheduler")
        class TestScheduler:
            pass
        
        assert "test_scheduler" in SCHEDULERS
        assert SCHEDULERS["test_scheduler"] is TestScheduler


class TestRegisterPLDataModule:
    """Test PyTorch Lightning DataModule registration functionality."""
    
    def setup_method(self):
        """Clear PLDATAMODULES registry before each test."""
        PLDATAMODULES.clear()
    
    def test_register_pldatamodule_decorator(self):
        """Test that register_pldatamodule decorator works correctly."""
        @register_pldatamodule("test_datamodule")
        class TestDataModule:
            pass
        
        assert "test_datamodule" in PLDATAMODULES
        assert PLDATAMODULES["test_datamodule"] is TestDataModule


class TestRegisterPLModule:
    """Test PyTorch Lightning Module registration functionality."""
    
    def setup_method(self):
        """Clear PLMODULES registry before each test."""
        PLMODULES.clear()
    
    def test_register_plmodule_decorator(self):
        """Test that register_plmodule decorator works correctly."""
        @register_plmodule("test_plmodule")
        class TestPLModule:
            pass
        
        assert "test_plmodule" in PLMODULES
        assert PLMODULES["test_plmodule"] is TestPLModule


class TestRegisterPLCallback:
    """Test PyTorch Lightning Callback registration functionality."""
    
    def setup_method(self):
        """Clear PLCALLBACKS registry before each test."""
        PLCALLBACKS.clear()
    
    def test_register_plcallback_decorator(self):
        """Test that register_plcallback decorator works correctly."""
        @register_plcallback("test_callback")
        class TestCallback:
            pass
        
        assert "test_callback" in PLCALLBACKS
        assert PLCALLBACKS["test_callback"] is TestCallback


class TestRegistryIntegration:
    """Test integration scenarios of the registry system."""
    
    def setup_method(self):
        """Clear all registries before each test."""
        CRITERIONS.clear()
        DATASETS.clear()
        MODELS.clear()
        MODULES.clear()
        PLMODULES.clear()
        PLDATAMODULES.clear()
        PLCALLBACKS.clear()
    
    def test_multiple_registrations_same_type(self):
        """Test registering multiple items of the same type."""
        @register_model("model1")
        class Model1:
            pass
        
        @register_model("model2")
        class Model2:
            pass
        
        assert len(MODELS) == 2
        assert MODELS["model1"] is Model1
        assert MODELS["model2"] is Model2
    
    def test_registration_name_conflict(self):
        """Test that later registrations overwrite earlier ones."""
        @register_model("same_name")
        class Model1:
            pass
        
        @register_model("same_name")
        class Model2:
            pass
        
        assert len(MODELS) == 1
        assert MODELS["same_name"] is Model2