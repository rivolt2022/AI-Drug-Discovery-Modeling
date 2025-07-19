"""
AI Drug Discovery Modeling - Source Package

이 패키지는 AI 신약개발 모델링을 위한 소스 코드를 포함합니다.
"""

from .data_loader import DrugDiscoveryDataLoader, create_data_loader

__version__ = "1.0.0"
__author__ = "AI Drug Discovery Team"

__all__ = [
    'DrugDiscoveryDataLoader',
    'create_data_loader'
] 