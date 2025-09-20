"""
Test script for DomainAwareTrainer with Amazon Bedrock and Azure OpenAI integration.
This script performs basic validation of the setup without making actual API calls.
"""

import os
import sys
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['openai', 'boto3', 'botocore']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} is NOT installed")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
    logger.error("Run: pip install .")
        return False
    
    return True

def check_environment_variables():
    """Check if required environment variables are set."""
    required_vars = {
        'AWS_ACCESS_KEY_ID': 'AWS access key for Bedrock',
        'AWS_SECRET_ACCESS_KEY': 'AWS secret key for Bedrock',
        'OPENAI_API_KEY': 'OpenAI API key (or AZURE_OPENAI_KEY for Azure)'
    }
    
    missing_vars = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            masked_value = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '*' * len(value)
            logger.info(f"✓ {var}: {masked_value}")
        else:
            missing_vars.append(f"{var} ({description})")
            logger.warning(f"✗ {var} is not set")
    
    if missing_vars:
        logger.warning("Missing environment variables:")
        for var in missing_vars:
            logger.warning(f"  - {var}")
    
    return len(missing_vars) == 0

def test_imports():
    """Test importing the main classes."""
    try:
        from ..finetuning_pipeline.DomainAwareTrainerBedrock import DomainAwareTrainer, FineTuningConfig
        logger.info("✓ Successfully imported DomainAwareTrainer and FineTuningConfig")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import main classes: {e}")
        return False

def test_configuration():
    """Test configuration setup."""
    try:
        from ..finetuning_pipeline.DomainAwareTrainerBedrock import FineTuningConfig
        
        # Test default configuration
        config = FineTuningConfig()
        logger.info(f"✓ Default model: {config.model}")
        logger.info(f"✓ Default epochs: {config.n_epochs}")
        logger.info(f"✓ Default batch size: {config.batch_size}")
        
        # Test custom configuration
        custom_config = FineTuningConfig(
            domain_name="test_domain",
            model="gpt-4-turbo-2024-04-09",
            n_epochs=5
        )
        logger.info(f"✓ Custom configuration created for domain: {custom_config.domain_name}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

def test_trainer_initialization():
    """Test trainer initialization without making API calls."""
    try:
        from DomainAwareTrainerBedrock import DomainAwareTrainer
        
        # Test with dummy credentials (won't make actual calls)
        trainer = DomainAwareTrainer(
            api_key="dummy-key",
            aws_access_key_id="dummy-access-key",
            aws_secret_access_key="dummy-secret-key",
            aws_region="us-east-1"
        )
        
        logger.info("✓ Trainer initialized successfully")
        logger.info(f"✓ Bedrock client available: {trainer.bedrock_client is not None}")
        logger.info(f"✓ OpenAI client available: {trainer.client is not None}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Trainer initialization failed: {e}")
        return False

def run_all_tests():
    """Run all validation tests."""
    logger.info("=" * 60)
    logger.info("DomainAwareTrainer Setup Validation")
    logger.info("=" * 60)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Environment Variables", check_environment_variables),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Trainer Initialization", test_trainer_initialization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Your setup is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Set your actual API credentials in environment variables")
        logger.info("2. Review and run example_usage.py")
        logger.info("3. Check SETUP.md for detailed usage instructions")
    else:
        logger.warning("⚠️  Some tests failed. Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
