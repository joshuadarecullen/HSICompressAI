#!/usr/bin/env python3
"""
Test PLModule and PLDataModule registry functionality
"""

print("=== Testing PLModule and PLDataModule Registry ===\n")

# Step 1: Check registry before imports
from hsicompressai.registry import PLMODULES, PLDATAMODULES

print("1. Registry state before imports:")
print(f"PLMODULES keys: {list(PLMODULES.keys())}")
print(f"PLDATAMODULES keys: {list(PLDATAMODULES.keys())}")

# Step 2: Import modules to trigger auto-registration
print(f"\n2. Importing modules to trigger registration...")

try:
    from hsicompressai.models.hsn11_module import HSN11LitModule
    print("✓ Imported HSN11LitModule")
except ImportError as e:
    print(f"✗ Failed to import HSN11LitModule: {e}")

try:
    from hsicompressai.datamodules.hyspecnet11kdatamodule import HySpecNet11kDataModule
    print("✓ Imported HySpecNet11kDataModule")
except ImportError as e:
    print(f"✗ Failed to import HySpecNet11kDataModule: {e}")

# Step 3: Check registry after imports
print(f"\n3. Registry state after imports:")
print(f"PLMODULES keys: {list(PLMODULES.keys())}")
print(f"PLDATAMODULES keys: {list(PLDATAMODULES.keys())}")

# Step 4: Test lookup functionality
print(f"\n4. Testing registry lookups:")

if "HySpecNet11k" in PLMODULES:
    plmodule_class = PLMODULES["HySpecNet11k"]
    print(f"✓ Found PLModule 'HySpecNet11k': {plmodule_class}")
    print(f"  Class name: {plmodule_class.__name__}")
    print(f"  Module: {plmodule_class.__module__}")
else:
    print("✗ PLModule 'HySpecNet11k' not found in registry")

if "HySpecNet11k" in PLDATAMODULES:
    pldatamodule_class = PLDATAMODULES["HySpecNet11k"]
    print(f"✓ Found PLDataModule 'HySpecNet11k': {pldatamodule_class}")
    print(f"  Class name: {pldatamodule_class.__name__}")
    print(f"  Module: {pldatamodule_class.__module__}")
else:
    print("✗ PLDataModule 'HySpecNet11k' not found in registry")

# Step 5: Test instantiation (without actual parameters to avoid complexity)
print(f"\n5. Testing class retrieval for instantiation:")

try:
    # Just verify we can get the class and inspect it
    if "HySpecNet11k" in PLMODULES:
        PLModuleClass = PLMODULES["HySpecNet11k"]
        print(f"✓ PLModule class retrieved: {PLModuleClass}")
        print(f"  Init signature: {PLModuleClass.__init__.__annotations__ if hasattr(PLModuleClass.__init__, '__annotations__') else 'No annotations'}")
        
    if "HySpecNet11k" in PLDATAMODULES:
        PLDataModuleClass = PLDATAMODULES["HySpecNet11k"]
        print(f"✓ PLDataModule class retrieved: {PLDataModuleClass}")
        print(f"  Init signature: {PLDataModuleClass.__init__.__annotations__ if hasattr(PLDataModuleClass.__init__, '__annotations__') else 'No annotations'}")

except Exception as e:
    print(f"✗ Error during class retrieval: {e}")

print(f"\n=== Test Complete ===")
print(f"Registry auto-registration: {'✓ WORKING' if PLMODULES or PLDATAMODULES else '✗ FAILED'}")