"""
Enhanced AI-based Failure Mode Classification Module
Provides intelligent failure code assignment using embeddings, SpaCy NLP, and rule-based expert systems
"""

import os
import json
import logging
import time
import re
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from collections import defaultdict

# AI/ML imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence transformers not available. Install with: pip install sentence-transformers")

try:
    import spacy
    from spacy.tokens import Doc
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("SpaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests not available. Install with: pip install requests")

# Suppress warnings
warnings.filterwarnings("ignore")

@dataclass
class AIClassificationResult:
    """Result of AI classification"""
    code: str
    description: str
    confidence: float
    method: str  # 'expert_system', 'contextual_patterns', 'temporal_analysis', 'ai_embeddings', 'ai_spacy', 'dictionary_fallback'
    matched_keyword: str = ''
    reasoning: str = ''
    entities: Optional[List[str]] = None  # SpaCy entities found
    equipment_type: str = ''    # Detected equipment type
    failure_indicators: Optional[List[str]] = None  # Failure-related terms

class ExpertSystemClassifier:
    """Rule-based expert system for failure mode classification"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.equipment_patterns = self._initialize_equipment_patterns()
        self.failure_patterns = self._initialize_failure_patterns()
    
    def _initialize_rules(self) -> List[Dict]:
        return [
            {
                'name': 'bearing_failure',
                'conditions': [
                    {'type': 'keyword', 'value': 'bearing', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'noise', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'vibration', 'weight': 0.2},
                    {'type': 'pattern', 'value': r'bearing.*fail', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'bearing.*noise', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'bearing.*vibration', 'weight': 0.3}
                ],
                'failure_code': 'Bearing Failure',
                'description': 'Bearing Failure'
            },
            {
                'name': 'seal_leak',
                'conditions': [
                    {'type': 'keyword', 'value': 'seal', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'leak', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'seal.*leak', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'packing.*leak', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'mechanical.*seal', 'weight': 0.3}
                ],
                'failure_code': 'Seal Leak',
                'description': 'Seal Leak'
            },
            {
                'name': 'motor_overheating',
                'conditions': [
                    {'type': 'keyword', 'value': 'motor', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'overheat', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'temperature', 'weight': 0.2},
                    {'type': 'pattern', 'value': r'motor.*overheat', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'temperature.*high', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'thermal.*trip', 'weight': 0.4}
                ],
                'failure_code': 'Motor Overheating',
                'description': 'Motor Overheating'
            },
            {
                'name': 'pump_cavitation',
                'conditions': [
                    {'type': 'keyword', 'value': 'pump', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'cavitation', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'suction', 'weight': 0.2},
                    {'type': 'pattern', 'value': r'cavitation', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'noise.*suction', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'vibration.*flow', 'weight': 0.2}
                ],
                'failure_code': 'Pump Cavitation',
                'description': 'Pump Cavitation'
            },
            {
                'name': 'electrical_fault',
                'conditions': [
                    {'type': 'keyword', 'value': 'electrical', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'fault', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'short', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'electrical.*fault', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'short.*circuit', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'ground.*fault', 'weight': 0.4}
                ],
                'failure_code': 'Electrical Fault',
                'description': 'Electrical Fault'
            },
            {
                'name': 'valve_stuck',
                'conditions': [
                    {'type': 'keyword', 'value': 'valve', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'stuck', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'seized', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'valve.*stuck', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'valve.*seized', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'valve.*not.*open', 'weight': 0.3}
                ],
                'failure_code': 'Valve Stuck/Seized',
                'description': 'Valve Stuck/Seized'
            },
            {
                'name': 'belt_failure',
                'conditions': [
                    {'type': 'keyword', 'value': 'belt', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'broken', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'slipping', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'belt.*break', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'belt.*slip', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'v.*belt', 'weight': 0.2}
                ],
                'failure_code': 'Belt Failure',
                'description': 'Belt Failure'
            },
            {
                'name': 'corrosion',
                'conditions': [
                    {'type': 'keyword', 'value': 'corrosion', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'rust', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'pitting', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'corrosion', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'rust.*formation', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'pitting.*corrosion', 'weight': 0.4}
                ],
                'failure_code': 'Corrosion',
                'description': 'Corrosion'
            },
            {
                'name': 'packing_leak',
                'conditions': [
                    {'type': 'keyword', 'value': 'packing', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'leak', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'gland', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'packing.*leak', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'gland.*leak', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'packing.*seal', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'stuffing.*box', 'weight': 0.3}
                ],
                'failure_code': 'Packing Leak',
                'description': 'Packing Leak'
            },
            {
                'name': 'lubrication_failure',
                'conditions': [
                    {'type': 'keyword', 'value': 'lubrication', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'lubricant', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'oil', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'grease', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'dry', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'lubrication.*fail', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'oil.*level.*low', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'grease.*not.*flow', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'dry.*running', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'no.*lubrication', 'weight': 0.5}
                ],
                'failure_code': 'Lubrication Failure',
                'description': 'Lubrication Failure'
            },
            {
                'name': 'signal_fault',
                'conditions': [
                    {'type': 'keyword', 'value': 'signal', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'indication', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'alarm', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'reading', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'display', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'sensor', 'weight': 0.2},
                    {'type': 'pattern', 'value': r'signal.*fault', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'signal.*error', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'false.*alarm', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'wrong.*reading', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'display.*error', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'indication.*wrong', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'sensor.*reading.*incorrect', 'weight': 0.5}
                ],
                'failure_code': 'Signal Fault/Indication Error',
                'description': 'Signal Fault/Indication Error'
            },
            # New expert rules based on training data analysis
            {
                'name': 'faulty_signal_indication_alarm',
                'conditions': [
                    {'type': 'keyword', 'value': 'malfunction', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'fault', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'alarm', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'tripped', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'error', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'noise', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'smoke', 'weight': 0.2},
                    {'type': 'pattern', 'value': r'faulty.*signal', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'false.*alarm', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'signal.*malfunction', 'weight': 0.5}
                ],
                'failure_code': 'Faulty Signal/Indication/Alarm',
                'description': 'Faulty Signal/Indication/Alarm'
            },
            {
                'name': 'leakage',
                'conditions': [
                    {'type': 'keyword', 'value': 'leaking', 'weight': 0.5},
                    {'type': 'keyword', 'value': 'leak', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'drip', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'seep', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'leak.*on', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'leaking.*from', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'seal.*leak', 'weight': 0.4}
                ],
                'failure_code': 'Leakage',
                'description': 'Leakage'
            },
            {
                'name': 'looseness',
                'conditions': [
                    {'type': 'keyword', 'value': 'loose', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'looseness', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'rupture', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'ruptured', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'not.*functioning.*properly', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'loose.*connection', 'weight': 0.4}
                ],
                'failure_code': 'Looseness',
                'description': 'Looseness'
            },
            {
                'name': 'contamination',
                'conditions': [
                    {'type': 'keyword', 'value': 'contamination', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'contaminated', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'emergency', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'troubleshoot', 'weight': 0.2},
                    {'type': 'pattern', 'value': r'oil.*analyzer', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'oil.*pressure', 'weight': 0.3}
                ],
                'failure_code': 'Contamination',
                'description': 'Contamination'
            },
            {
                'name': 'control_failure',
                'conditions': [
                    {'type': 'keyword', 'value': 'control', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'issue', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'tracking', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'limit', 'weight': 0.2},
                    {'type': 'pattern', 'value': r'control.*failure', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'not.*meeting.*limit', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'valve.*tracking', 'weight': 0.4}
                ],
                'failure_code': 'Control failure',
                'description': 'Control failure'
            },
            {
                'name': 'calibration',
                'conditions': [
                    {'type': 'keyword', 'value': 'calibration', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'calibrated', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'troubleshot', 'weight': 0.2},
                    {'type': 'pattern', 'value': r'need.*to.*be.*calibrated', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'calibration.*error', 'weight': 0.4}
                ],
                'failure_code': 'Calibration',
                'description': 'Calibration'
            },
            {
                'name': 'power_supply',
                'conditions': [
                    {'type': 'keyword', 'value': 'power', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'tripped', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'emergency', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'noise', 'weight': 0.2},
                    {'type': 'pattern', 'value': r'power.*supply', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'reconnect.*power', 'weight': 0.4}
                ],
                'failure_code': 'Power Supply',
                'description': 'Power Supply'
            },
            {
                'name': 'open_circuit',
                'conditions': [
                    {'type': 'keyword', 'value': 'broken', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'open', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'circuit', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'open.*circuit', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'broken.*connection', 'weight': 0.4}
                ],
                'failure_code': 'Open circuit',
                'description': 'Open circuit'
            },
            {
                'name': 'seal',
                'conditions': [
                    {'type': 'keyword', 'value': 'seal', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'leak', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'alarm', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'error', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'alarming', 'weight': 0.2},
                    {'type': 'pattern', 'value': r'seal.*leak', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'mechanical.*seal', 'weight': 0.4}
                ],
                'failure_code': 'Seal',
                'description': 'Seal'
            },
            {
                'name': 'overheating',
                'conditions': [
                    {'type': 'keyword', 'value': 'overheating', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'overheat', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'tripped', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'hot', 'weight': 0.2},
                    {'type': 'pattern', 'value': r'overheating', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'temperature.*high', 'weight': 0.4}
                ],
                'failure_code': 'Overheating',
                'description': 'Overheating'
            },
            {
                'name': 'vibration_excessive',
                'conditions': [
                    {'type': 'keyword', 'value': 'vibration', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'vibrating', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'excessive', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'abnormal', 'weight': 0.2},
                    {'type': 'pattern', 'value': r'excessive.*vibration', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'vibration.*high', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'abnormal.*vibration', 'weight': 0.4}
                ],
                'failure_code': 'Vibration',
                'description': 'Vibration'
            },
            {
                'name': 'electrical_short_circuit',
                'conditions': [
                    {'type': 'keyword', 'value': 'short', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'circuit', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'electrical', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'shorted', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'short.*circuit', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'electrical.*short', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'circuit.*short', 'weight': 0.4}
                ],
                'failure_code': 'Short Circuiting',
                'description': 'Short Circuiting'
            },
            {
                'name': 'lubrication_failure',
                'conditions': [
                    {'type': 'keyword', 'value': 'lubrication', 'weight': 0.4},
                    {'type': 'keyword', 'value': 'lubricant', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'oil', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'grease', 'weight': 0.2},
                    {'type': 'keyword', 'value': 'dry', 'weight': 0.3},
                    {'type': 'pattern', 'value': r'lubrication.*fail', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'oil.*level.*low', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'grease.*not.*flow', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'dry.*running', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'no.*lubrication', 'weight': 0.5}
                ],
                'failure_code': 'Lubrication Failure',
                'description': 'Lubrication Failure'
            },
            {
                'name': 'sensor_failure',
                'conditions': [
                    {'type': 'keyword', 'value': 'sensor', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'transmitter', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'faulty', 'weight': 0.3},
                    {'type': 'keyword', 'value': 'error', 'weight': 0.2},
                    {'type': 'pattern', 'value': r'sensor.*fail', 'weight': 0.5},
                    {'type': 'pattern', 'value': r'transmitter.*fail', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'faulty.*sensor', 'weight': 0.4},
                    {'type': 'pattern', 'value': r'sensor.*error', 'weight': 0.4}
                ],
                'failure_code': 'Sensor Failure',
                'description': 'Sensor Failure'
            }
        ]
    
    def _initialize_equipment_patterns(self) -> Dict:
        return {
            'pump': {
                'keywords': ['pump', 'impeller', 'suction', 'discharge', 'flow', 'pressure', 'NPSH'],
                'common_failures': ['cavitation', 'seal leak', 'bearing failure', 'impeller damage']
            },
            'motor': {
                'keywords': ['motor', 'current', 'voltage', 'temperature', 'winding', 'insulation'],
                'common_failures': ['overheating', 'bearing failure', 'electrical fault', 'insulation failure']
            },
            'valve': {
                'keywords': ['valve', 'actuator', 'position', 'stroke', 'seat', 'disc'],
                'common_failures': ['stuck', 'leak', 'actuator failure', 'seat damage']
            },
            'compressor': {
                'keywords': ['compressor', 'pressure', 'discharge', 'suction', 'intercooler'],
                'common_failures': ['overheating', 'vibration', 'bearing failure', 'seal leak']
            },
            'fan': {
                'keywords': ['fan', 'blade', 'airflow', 'duct', 'ventilation'],
                'common_failures': ['blade damage', 'bearing failure', 'vibration', 'imbalance']
            }
        }
    
    def _initialize_failure_patterns(self) -> Dict:
        return {
            'bearing_failure': [r'bearing.*noise', r'bearing.*vibration', r'bearing.*fail', r'bearing.*worn'],
            'seal_leak': [r'seal.*leak', r'packing.*leak', r'mechanical.*seal', r'gland.*leak'],
            'packing_leak': [r'packing.*leak', r'gland.*leak', r'packing.*seal', r'stuffing.*box'],
            'lubrication_failure': [r'lubrication.*fail', r'oil.*level.*low', r'grease.*not.*flow', r'dry.*running', r'no.*lubrication'],
            'signal_fault': [r'signal.*fault', r'signal.*error', r'false.*alarm', r'wrong.*reading', r'display.*error', r'indication.*wrong', r'sensor.*reading.*incorrect'],
            'overheating': [r'overheat', r'temperature.*high', r'thermal.*trip', r'hot.*running'],
            'electrical_fault': [r'electrical.*fault', r'short.*circuit', r'ground.*fault', r'electrical.*leak'],
            'vibration': [r'vibration', r'excessive.*vibration', r'vibration.*high', r'vibration.*alarm'],
            'noise': [r'noise', r'excessive.*noise', r'unusual.*noise', r'noise.*level']
        }
    
    def classify(self, description: str) -> Tuple[str, str, float]:
        """Classify failure mode using expert system rules"""
        if not description:
            return ('No Failure Mode Identified', 'No Failure Mode Identified', 0.0)
        
        description_lower = description.lower()
        scores = {}
        
        for rule in self.rules:
            score = 0.0
            matched_conditions = 0
            
            for condition in rule['conditions']:
                if condition['type'] == 'keyword':
                    if condition['value'].lower() in description_lower:
                        score += condition['weight']
                        matched_conditions += 1
                elif condition['type'] == 'pattern':
                    if re.search(condition['value'], description_lower, re.IGNORECASE):
                        score += condition['weight']
                        matched_conditions += 1
            
            # Use raw score instead of normalizing by total conditions
            if matched_conditions > 0:
                # Boost confidence based on number of matched conditions
                score = min(1.0, score * (1 + matched_conditions * 0.2))
                scores[rule['failure_code']] = {
                    'score': score,
                    'description': rule['description'],
                    'matched_conditions': matched_conditions
                }
        
        if scores:
            best_code = max(scores.keys(), key=lambda k: scores[k]['score'])
            return (best_code, 
                   scores[best_code]['description'], 
                   scores[best_code]['score'])
        
        return ('No Failure Mode Identified', 'No Failure Mode Identified', 0.0)

class ContextualPatternClassifier:
    """Contextual pattern recognition for equipment-specific failures"""
    
    def __init__(self):
        self.equipment_contexts = {
            'pump': {
                'common_failures': ['cavitation', 'seal leak', 'packing leak', 'bearing failure', 'impeller damage', 'lubrication failure'],
                'context_words': ['suction', 'discharge', 'flow', 'pressure', 'NPSH'],
                'failure_patterns': {
                    'cavitation': [r'cavitation', r'noise.*suction', r'vibration.*flow'],
                    'seal_leak': [r'seal.*leak', r'packing.*leak', r'mechanical.*seal'],
                    'packing_leak': [r'packing.*leak', r'gland.*leak', r'stuffing.*box'],
                    'bearing_failure': [r'bearing.*noise', r'bearing.*vibration', r'bearing.*fail'],
                    'lubrication_failure': [r'lubrication.*fail', r'oil.*level.*low', r'dry.*running']
                }
            },
            'motor': {
                'common_failures': ['overheating', 'bearing failure', 'electrical fault', 'insulation failure', 'Bearing', 'Control failure'],
                'context_words': ['current', 'voltage', 'temperature', 'winding', 'insulation', 'motor'],
                'failure_patterns': {
                    'overheating': [r'motor.*overheat', r'temperature.*high', r'thermal.*trip'],
                    'electrical_fault': [r'electrical.*fault', r'short.*circuit', r'ground.*fault'],
                    'insulation_failure': [r'insulation.*fail', r'winding.*short', r'electrical.*leak'],
                    'bearing_failure': [r'bearing.*noise', r'bearing.*vibration', r'bearing.*fail'],
                    'change_motor': [r'change.*motor', r'replace.*motor', r'motor.*change']
                }
            },
            'control_valve': {
                'common_failures': ['stuck', 'leak', 'actuator failure', 'positioner failure', 'signal fault', 'calibration error'],
                'context_words': ['actuator', 'positioner', 'signal', 'calibration', 'control', 'automatic', 'feedback'],
                'failure_patterns': {
                    'stuck': [r'control.*valve.*stuck', r'valve.*not.*respond', r'position.*not.*change'],
                    'actuator_failure': [r'actuator.*fail', r'actuator.*not.*work', r'positioner.*fail'],
                    'signal_fault': [r'signal.*fault', r'positioner.*signal', r'control.*signal.*error'],
                    'calibration_error': [r'calibration.*error', r'position.*wrong', r'valve.*not.*calibrated']
                }
            },
            'compressor': {
                'common_failures': ['overheating', 'vibration', 'bearing failure', 'seal leak'],
                'context_words': ['pressure', 'discharge', 'suction', 'intercooler'],
                'failure_patterns': {
                    'overheating': [r'compressor.*overheat', r'discharge.*temperature', r'thermal.*trip'],
                    'vibration': [r'compressor.*vibration', r'excessive.*vibration', r'vibration.*alarm'],
                    'bearing_failure': [r'bearing.*noise', r'bearing.*vibration', r'bearing.*fail']
                }
            },
            'turbine': {
                'common_failures': ['vibration', 'bearing failure', 'blade damage', 'steam leak', 'governor failure'],
                'context_words': ['steam', 'blade', 'rotor', 'governor', 'turbine', 'rpm', 'power'],
                'failure_patterns': {
                    'vibration': [r'turbine.*vibration', r'rotor.*vibration', r'excessive.*vibration'],
                    'blade_damage': [r'blade.*damage', r'blade.*bent', r'blade.*break'],
                    'steam_leak': [r'steam.*leak', r'steam.*seal', r'steam.*packing'],
                    'governor_failure': [r'governor.*fail', r'speed.*control', r'rpm.*control']
                }
            },
            'vessel': {
                'common_failures': ['corrosion', 'leak', 'pressure fault', 'level fault', 'temperature fault'],
                'context_words': ['tank', 'pressure', 'level', 'temperature', 'vessel', 'storage'],
                'failure_patterns': {
                    'corrosion': [r'vessel.*corrosion', r'tank.*corrosion', r'rust.*formation'],
                    'leak': [r'vessel.*leak', r'tank.*leak', r'seam.*leak'],
                    'pressure_fault': [r'pressure.*fault', r'pressure.*high', r'pressure.*low'],
                    'level_fault': [r'level.*fault', r'level.*high', r'level.*low']
                }
            },
            'agitator': {
                'common_failures': ['bearing failure', 'seal leak', 'vibration', 'impeller damage', 'motor failure'],
                'context_words': ['impeller', 'mixing', 'agitation', 'shaft', 'gear'],
                'failure_patterns': {
                    'bearing_failure': [r'agitator.*bearing', r'bearing.*noise', r'bearing.*vibration'],
                    'seal_leak': [r'agitator.*seal', r'seal.*leak', r'mechanical.*seal'],
                    'impeller_damage': [r'impeller.*damage', r'impeller.*bent', r'impeller.*break'],
                    'vibration': [r'agitator.*vibration', r'excessive.*vibration', r'vibration.*alarm']
                }
            },
            'mixer': {
                'common_failures': ['bearing failure', 'seal leak', 'vibration', 'impeller damage', 'motor failure'],
                'context_words': ['impeller', 'mixing', 'agitation', 'shaft', 'gear'],
                'failure_patterns': {
                    'bearing_failure': [r'mixer.*bearing', r'bearing.*noise', r'bearing.*vibration'],
                    'seal_leak': [r'mixer.*seal', r'seal.*leak', r'mechanical.*seal'],
                    'impeller_damage': [r'impeller.*damage', r'impeller.*bent', r'impeller.*break'],
                    'vibration': [r'mixer.*vibration', r'excessive.*vibration', r'vibration.*alarm']
                }
            },
            'transmitter': {
                'common_failures': ['signal fault', 'calibration error', 'sensor failure', 'communication fault', 'Faulty Signal/Indication/Alarm', 'Calibration'],
                'context_words': ['pressure', 'temperature', 'level', 'flow', 'signal', 'calibration', 'transmitter', 'freezing'],
                'failure_patterns': {
                    'signal_fault': [r'transmitter.*signal', r'signal.*fault', r'signal.*error'],
                    'calibration_error': [r'transmitter.*calibration', r'calibration.*error', r'need.*to.*be.*calibrated'],
                    'freezing': [r'transmitter.*freezing', r'freezing.*up', r'frozen.*transmitter'],
                    'faulty_signal': [r'faulty.*signal', r'false.*alarm', r'wrong.*reading']
                }
            },
            'valve': {
                'common_failures': ['stuck', 'leak', 'actuator failure', 'seat damage', 'valve_leak', 'valve_stuck', 'Control failure'],
                'context_words': ['actuator', 'position', 'stroke', 'seat', 'disc', 'valve', 'limit', 'switch'],
                'failure_patterns': {
                    'stuck': [r'valve.*stuck', r'valve.*seized', r'valve.*not.*open', r'valve.*would.*not.*open'],
                    'leak': [r'valve.*leak', r'seat.*leak', r'disc.*leak', r'leak.*from.*valve'],
                    'actuator_failure': [r'actuator.*fail', r'actuator.*not.*work', r'position.*not.*change'],
                    'control_failure': [r'valve.*tracking', r'not.*meeting.*limit', r'control.*failure']
                }
            },
            'pump': {
                'common_failures': ['cavitation', 'seal leak', 'packing leak', 'bearing failure', 'impeller damage', 'lubrication failure', 'Cavitation', 'Leakage'],
                'context_words': ['suction', 'discharge', 'flow', 'pressure', 'NPSH', 'pump', 'circulation'],
                'failure_patterns': {
                    'cavitation': [r'cavitation', r'noise.*suction', r'vibration.*flow', r'pump.*cavitation'],
                    'seal_leak': [r'seal.*leak', r'packing.*leak', r'mechanical.*seal', r'pump.*seal'],
                    'packing_leak': [r'packing.*leak', r'gland.*leak', r'stuffing.*box'],
                    'bearing_failure': [r'bearing.*noise', r'bearing.*vibration', r'bearing.*fail'],
                    'lubrication_failure': [r'lubrication.*fail', r'oil.*level.*low', r'dry.*running'],
                    'circulation': [r'circulation.*pump', r'pump.*needed', r'basin.*pump']
                }
            },
            'fan': {
                'common_failures': ['blade damage', 'bearing failure', 'vibration', 'imbalance', 'Bearing', 'Contamination', 'Overheating'],
                'context_words': ['blade', 'airflow', 'duct', 'ventilation', 'fan', 'combustion'],
                'failure_patterns': {
                    'blade_damage': [r'blade.*damage', r'blade.*bent', r'blade.*break', r'fan.*blade'],
                    'bearing_failure': [r'bearing.*noise', r'bearing.*vibration', r'bearing.*fail'],
                    'vibration': [r'fan.*vibration', r'excessive.*vibration', r'vibration.*alarm'],
                    'contamination': [r'fan.*contamination', r'contamination.*fan', r'combustion.*fan']
                }
            },
            'motor': {
                'common_failures': ['overheating', 'bearing failure', 'electrical fault', 'insulation failure', 'Bearing', 'Control failure'],
                'context_words': ['current', 'voltage', 'temperature', 'winding', 'insulation', 'motor'],
                'failure_patterns': {
                    'overheating': [r'motor.*overheat', r'temperature.*high', r'thermal.*trip'],
                    'electrical_fault': [r'electrical.*fault', r'short.*circuit', r'ground.*fault'],
                    'insulation_failure': [r'insulation.*fail', r'winding.*short', r'electrical.*leak'],
                    'bearing_failure': [r'bearing.*noise', r'bearing.*vibration', r'bearing.*fail'],
                    'change_motor': [r'change.*motor', r'replace.*motor', r'motor.*change']
                }
            },
            'reactor': {
                'common_failures': ['temperature fault', 'pressure fault', 'agitation failure', 'cooling failure'],
                'context_words': ['reaction', 'temperature', 'pressure', 'catalyst', 'cooling'],
                'failure_patterns': {
                    'temperature_fault': [r'temperature.*fault', r'temperature.*high', r'temperature.*low'],
                    'pressure_fault': [r'pressure.*fault', r'pressure.*high', r'pressure.*low'],
                    'agitation_failure': [r'agitation.*fail', r'mixing.*fail', r'stirring.*fail'],
                    'cooling_failure': [r'cooling.*fail', r'cooling.*not.*work', r'temperature.*control']
                }
            },
            'boiler': {
                'common_failures': ['water level fault', 'pressure fault', 'burner failure', 'tube leak', 'feedwater fault'],
                'context_words': ['steam', 'water', 'burner', 'tube', 'feedwater', 'drum'],
                'failure_patterns': {
                    'water_level_fault': [r'water.*level.*fault', r'level.*high', r'level.*low'],
                    'pressure_fault': [r'boiler.*pressure', r'pressure.*high', r'pressure.*low'],
                    'burner_failure': [r'burner.*fail', r'burner.*not.*light', r'combustion.*fail'],
                    'tube_leak': [r'tube.*leak', r'tube.*rupture', r'tube.*failure']
                }
            },
            'generator': {
                'common_failures': ['electrical fault', 'bearing failure', 'cooling failure', 'excitation fault'],
                'context_words': ['electrical', 'power', 'voltage', 'current', 'excitation', 'cooling'],
                'failure_patterns': {
                    'electrical_fault': [r'electrical.*fault', r'power.*fault', r'voltage.*fault'],
                    'bearing_failure': [r'generator.*bearing', r'bearing.*noise', r'bearing.*vibration'],
                    'cooling_failure': [r'cooling.*fail', r'cooling.*not.*work', r'temperature.*high'],
                    'excitation_fault': [r'excitation.*fault', r'excitation.*fail', r'voltage.*regulation']
                }
            },
            'exchanger': {
                'common_failures': ['tube leak', 'fouling', 'pressure drop', 'temperature fault'],
                'context_words': ['tube', 'shell', 'heat', 'temperature', 'pressure', 'fouling'],
                'failure_patterns': {
                    'tube_leak': [r'tube.*leak', r'tube.*rupture', r'tube.*failure'],
                    'fouling': [r'fouling', r'tube.*fouling', r'shell.*fouling'],
                    'pressure_drop': [r'pressure.*drop', r'pressure.*high', r'pressure.*drop'],
                    'temperature_fault': [r'temperature.*fault', r'temperature.*high', r'temperature.*low']
                }
            },
            'fan': {
                'common_failures': ['blade damage', 'bearing failure', 'vibration', 'imbalance'],
                'context_words': ['blade', 'airflow', 'duct', 'ventilation'],
                'failure_patterns': {
                    'blade_damage': [r'blade.*damage', r'blade.*bent', r'blade.*break'],
                    'bearing_failure': [r'bearing.*noise', r'bearing.*vibration', r'bearing.*fail'],
                    'imbalance': [r'imbalance', r'unbalance', r'vibration.*imbalance']
                }
            },
            'heat_exchanger': {
                'common_failures': ['tube leak', 'fouling', 'pressure drop', 'temperature fault'],
                'context_words': ['tube', 'shell', 'heat', 'temperature', 'pressure', 'fouling', 'exchanger'],
                'failure_patterns': {
                    'tube_leak': [r'tube.*leak', r'tube.*rupture', r'tube.*failure'],
                    'fouling': [r'fouling', r'tube.*fouling', r'shell.*fouling'],
                    'pressure_drop': [r'pressure.*drop', r'pressure.*high', r'pressure.*drop'],
                    'temperature_fault': [r'temperature.*fault', r'temperature.*high', r'temperature.*low']
                }
            },
            'conveyor': {
                'common_failures': ['belt damage', 'bearing failure', 'drive failure', 'alignment issue'],
                'context_words': ['belt', 'conveyor', 'drive', 'pulley', 'idler', 'material handling'],
                'failure_patterns': {
                    'belt_damage': [r'belt.*damage', r'belt.*tear', r'belt.*break'],
                    'bearing_failure': [r'bearing.*noise', r'bearing.*vibration', r'bearing.*fail'],
                    'drive_failure': [r'drive.*fail', r'motor.*fail', r'gearbox.*fail'],
                    'alignment_issue': [r'alignment.*issue', r'misalignment', r'pulley.*misaligned']
                }
            },
            'transformer': {
                'common_failures': ['electrical fault', 'overheating', 'insulation failure', 'oil leak'],
                'context_words': ['transformer', 'electrical', 'voltage', 'current', 'oil', 'insulation'],
                'failure_patterns': {
                    'electrical_fault': [r'electrical.*fault', r'voltage.*fault', r'current.*fault'],
                    'overheating': [r'transformer.*overheat', r'temperature.*high', r'thermal.*trip'],
                    'insulation_failure': [r'insulation.*fail', r'winding.*short', r'electrical.*leak'],
                    'oil_leak': [r'oil.*leak', r'transformer.*leak', r'oil.*level.*low']
                }
            }
        }
    
    def detect_equipment_context(self, description: str) -> str:
        """Detect equipment type from description"""
        desc_lower = description.lower()
        
        for equipment, context in self.equipment_contexts.items():
            if equipment in desc_lower:
                return equipment
            
            # Check for context words
            context_matches = sum(1 for word in context['context_words'] 
                                if word in desc_lower)
            if context_matches >= 2:
                return equipment
        
        return 'unknown'
    
    def classify_with_context(self, description: str) -> Tuple[str, str, float, str]:
        """Classify failure mode considering equipment context"""
        equipment = self.detect_equipment_context(description)
        
        if equipment == 'unknown':
            return ('No Failure Mode Identified', 'No Failure Mode Identified', 0.0, equipment)
        
        context = self.equipment_contexts[equipment]
        scores = {}
        
        for failure_type, patterns in context['failure_patterns'].items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, description, re.IGNORECASE):
                    score += 0.5
            
            # Bonus for equipment-specific keywords
            for keyword in context['common_failures']:
                if keyword in description.lower():
                    score += 0.3
            
            scores[failure_type] = min(1.0, score)
        
        if scores:
            best_failure = max(scores.keys(), key=scores.get)
            return (f"{equipment}_{best_failure}", 
                   f"{equipment.title()} {best_failure.replace('_', ' ').title()}", 
                   scores[best_failure],
                   equipment)
        
        return ('No Failure Mode Identified', 'No Failure Mode Identified', 0.0, equipment)

class TimeSeriesPatternClassifier:
    """Time-series pattern analysis for failure prediction"""
    
    def __init__(self, time_window_days: int = 30):
        self.time_window = time_window_days
        self.failure_history = {}
        self.seasonal_patterns = {}
        self.has_history = False
    
    def analyze_temporal_patterns(self, work_orders_df: pd.DataFrame):
        """Analyze failure patterns over time"""
        if work_orders_df.empty:
            return
        
        try:
            # Group by equipment and failure type
            equipment_failures = work_orders_df.groupby(['Equipment #', 'Failure Code']).agg({
                'Reported Date': 'count',
                'Description': list
            }).reset_index()
            
            # Identify recurring patterns
            for _, row in equipment_failures.iterrows():
                equipment = row['Equipment #']
                failure_code = row['Failure Code']
                failure_count = row['Reported Date']
                
                if failure_count > 1:
                    key = f"{equipment}_{failure_code}"
                    self.failure_history[key] = {
                        'count': failure_count,
                        'descriptions': row['Description'],
                        'pattern_strength': failure_count / len(work_orders_df)
                    }
            
            self.has_history = len(self.failure_history) > 0
            logging.info(f"Analyzed temporal patterns for {len(self.failure_history)} failure types")
            
        except Exception as e:
            logging.error(f"Error analyzing temporal patterns: {e}")
    
    def classify_with_temporal_context(self, description: str, equipment: str, 
                                     current_date: datetime) -> Tuple[str, str, float]:
        """Classify considering temporal patterns"""
        
        if not self.has_history:
            return ('No Failure Mode Identified', 'No Failure Mode Identified', 0.0)
        
        # Check for historical patterns
        temporal_scores = {}
        
        for pattern_key, pattern_data in self.failure_history.items():
            if equipment in pattern_key:
                failure_code = pattern_key.split('_')[1]
                
                # Check if description matches historical descriptions
                similarity_score = self._calculate_description_similarity(
                    description, pattern_data['descriptions']
                )
                
                # Combine with temporal pattern strength
                temporal_score = similarity_score * pattern_data['pattern_strength']
                temporal_scores[failure_code] = temporal_score
        
        if temporal_scores:
            best_code = max(temporal_scores.keys(), key=temporal_scores.get)
            return (best_code, 
                   f"Temporal Pattern Match: {best_code}", 
                   temporal_scores[best_code])
        
        return ('No Temporal Pattern Found', 'No Temporal Pattern Found', 0.0)
    
    def _calculate_description_similarity(self, new_desc: str, historical_descs: List[str]) -> float:
        """Calculate similarity between new and historical descriptions"""
        if not historical_descs:
            return 0.0
        
        try:
            from rapidfuzz import fuzz
            similarities = []
            for hist_desc in historical_descs:
                # Use fuzzy matching for similarity
                similarity = fuzz.partial_ratio(new_desc.lower(), hist_desc.lower()) / 100.0
                similarities.append(similarity)
            
            return max(similarities) if similarities else 0.0
        except ImportError:
            # Fallback to simple keyword matching
            new_words = set(new_desc.lower().split())
            max_similarity = 0.0
            for hist_desc in historical_descs:
                hist_words = set(hist_desc.lower().split())
                if new_words and hist_words:
                    similarity = len(new_words.intersection(hist_words)) / len(new_words.union(hist_words))
                    max_similarity = max(max_similarity, similarity)
            return max_similarity

class AIClassifier:
    """Enhanced AI-based failure mode classifier with expert systems and contextual analysis"""
    
    def __init__(self, 
                 cache_file: str = "ai_classification_cache.json",
                 confidence_threshold: float = 0.3,
                 spacy_model: str = "en_core_web_sm"):
        
        self.cache_file = cache_file
        self.confidence_threshold = confidence_threshold
        self.spacy_model = spacy_model
        
        # Initialize components
        self.embedding_model = None
        self.nlp = None
        self.cache = self._load_cache()
        self.failure_codes = {}
        self.failure_embeddings = {}
        
        # Initialize new classification methods
        self.expert_system = ExpertSystemClassifier()
        self.contextual_patterns = ContextualPatternClassifier()
        self.temporal_analysis = TimeSeriesPatternClassifier()
        
        # Equipment and failure indicators for SpaCy analysis
        self.equipment_types = {
            # Rotating equipment
            'pump', 'motor', 'compressor', 'fan', 'blower', 'turbine', 'generator', 'engine',
            'centrifugal pump', 'positive displacement pump', 'reciprocating pump', 'gear pump',
            'screw pump', 'diaphragm pump', 'submersible pump', 'vertical pump',
            'induction motor', 'synchronous motor', 'dc motor', 'servo motor',
            'reciprocating compressor', 'screw compressor', 'centrifugal compressor',
            'axial fan', 'centrifugal fan', 'propeller fan',
            
            # Valves and actuators
            'valve', 'actuator', 'control valve', 'ball valve', 'gate valve', 'globe valve',
            'butterfly valve', 'check valve', 'safety valve', 'relief valve', 'solenoid valve',
            'pneumatic actuator', 'hydraulic actuator', 'electric actuator', 'manual actuator',
            'positioner', 'limit switch', 'position indicator',
            
            # Instrumentation
            'sensor', 'transmitter', 'controller', 'switch', 'transducer', 'probe',
            'pressure transmitter', 'temperature transmitter', 'level transmitter', 'flow transmitter',
            'pressure sensor', 'temperature sensor', 'level sensor', 'flow sensor',
            'rtd', 'thermocouple', 'pressure gauge', 'temperature gauge', 'level gauge',
            'flow meter', 'orifice plate', 'venturi', 'rotameter', 'ultrasonic flow meter',
            
            # Mechanical components
            'bearing', 'seal', 'coupling', 'belt', 'chain', 'gear', 'shaft', 'impeller',
            'ball bearing', 'roller bearing', 'thrust bearing', 'journal bearing',
            'mechanical seal', 'packing seal', 'labyrinth seal', 'lip seal',
            'flexible coupling', 'rigid coupling', 'gear coupling', 'chain coupling',
            'v belt', 'flat belt', 'timing belt', 'roller chain', 'silent chain',
            'spur gear', 'helical gear', 'bevel gear', 'worm gear',
            
            # Vessels and tanks
            'tank', 'vessel', 'reactor', 'separator', 'clarifier', 'thickener',
            'storage tank', 'process tank', 'mixing tank', 'holding tank',
            'pressure vessel', 'heat exchanger', 'condenser', 'evaporator', 'reboiler',
            'shell and tube', 'plate heat exchanger', 'air cooled heat exchanger',
            'cooling tower', 'chiller', 'heater', 'furnace', 'boiler',
            
            # Filtration and separation
            'filter', 'strainer', 'separator', 'centrifuge', 'cyclone', 'hydrocyclone',
            'bag filter', 'cartridge filter', 'screen filter', 'magnetic separator',
            'gravity separator', 'coalescer', 'demister', 'mist eliminator',
            
            # Mixing and agitation
            'mixer', 'agitator', 'stirrer', 'blender', 'homogenizer', 'emulsifier',
            'propeller mixer', 'turbine mixer', 'anchor mixer', 'helical mixer',
            
            # Conveying and handling
            'conveyor', 'elevator', 'hopper', 'feeder', 'screw conveyor', 'belt conveyor',
            'bucket elevator', 'vibratory feeder', 'rotary feeder', 'pneumatic conveyor',
            
            # Electrical and control
            'transformer', 'switchgear', 'circuit breaker', 'fuse', 'relay', 'contactor',
            'vfd', 'variable frequency drive', 'inverter', 'ups', 'uninterruptible power supply',
            'plc', 'programmable logic controller', 'scada', 'distributed control system',
            'hmi', 'human machine interface', 'rtu', 'remote terminal unit',
            
            # Safety and protection
            'safety valve', 'relief valve', 'rupture disk', 'flame arrester', 'explosion vent',
            'emergency shutdown', 'esd', 'fire suppression', 'gas detection',
            
            # Utilities and support
            'cooling water', 'chilled water', 'steam', 'air', 'nitrogen', 'instrument air',
            'hydraulic system', 'pneumatic system', 'lubrication system', 'cooling system'
        }
        
        self.failure_indicators = {
            # General failure terms
            'failure', 'failed', 'broken', 'damaged', 'worn', 'defective', 'faulty',
            'malfunction', 'malfunctioning', 'inoperative', 'non functional', 'not working',
            'out of service', 'down', 'shutdown', 'stopped', 'inactive',
            
            # Physical damage
            'crack', 'cracked', 'rupture', 'ruptured', 'fracture', 'fractured',
            'break', 'broken', 'bent', 'deformed', 'distorted', 'twisted',
            'dented', 'scratched', 'gouged', 'pitted', 'scored', 'worn out',
            'eroded', 'corroded', 'rusted', 'oxidized', 'tarnished',
            
            # Leakage and fluid issues
            'leak', 'leaking', 'leaked', 'seep', 'seepage', 'drip', 'dripping',
            'spill', 'spillage', 'overflow', 'flood', 'flooding',
            'dry', 'dried out', 'no lubrication', 'insufficient lubrication',
            
            # Temperature and thermal issues
            'overheat', 'overheated', 'overheating', 'hot', 'burning', 'burnt',
            'thermal', 'temperature high', 'temperature low', 'thermal trip',
            'cold', 'frozen', 'freezing', 'thermal stress', 'thermal expansion',
            
            # Mechanical issues
            'seized', 'stuck', 'jammed', 'binding', 'not moving', 'immobile',
            'loose', 'loosened', 'detached', 'disconnected', 'unsecured',
            'vibration', 'vibrating', 'excessive vibration', 'abnormal vibration',
            'noise', 'noisy', 'excessive noise', 'unusual noise', 'rattle',
            'imbalance', 'unbalanced', 'misalignment', 'misaligned',
            
            # Electrical issues
            'electrical fault', 'electrical failure', 'short circuit', 'shorted',
            'open circuit', 'broken circuit', 'ground fault', 'earth fault',
            'arc', 'arcing', 'spark', 'sparking', 'electrical leak',
            'insulation failure', 'winding short', 'voltage drop', 'current high',
            'power loss', 'power failure', 'electrical trip',
            
            # Control and signal issues
            'signal fault', 'signal error', 'false alarm', 'wrong reading',
            'incorrect reading', 'calibration error', 'out of calibration',
            'control failure', 'control error', 'position error', 'tracking error',
            'communication fault', 'communication error', 'network fault',
            
            # Operational issues
            'not starting', 'not running', 'not operating', 'not functioning',
            'performance degradation', 'efficiency loss', 'capacity reduction',
            'flow restriction', 'pressure drop', 'pressure high', 'pressure low',
            'level high', 'level low', 'temperature fault', 'temperature error',
            
            # Safety and emergency
            'emergency', 'critical', 'urgent', 'dangerous', 'unsafe',
            'safety trip', 'safety shutdown', 'emergency shutdown',
            'alarm', 'alarming', 'tripped', 'trip', 'shutdown',
            'smoke', 'fire', 'explosion', 'burst', 'exploded',
            
            # Wear and deterioration
            'wear', 'worn', 'abraded', 'thinning', 'scuffed', 'surface loss',
            'pitting', 'scoring', 'fatigue', 'stress fracture', 'cyclic failure',
            'aging', 'deterioration', 'degradation', 'weakening',
            
            # Contamination and fouling
            'contamination', 'contaminated', 'fouling', 'fouled', 'clogged',
            'blocked', 'plugged', 'obstruction', 'restriction', 'dirt',
            'debris', 'foreign material', 'impurities', 'scale', 'scaling',
            
            # Operational problems
            'problem', 'issue', 'trouble', 'difficulty', 'concern',
            'not meeting specification', 'not meeting requirement',
            'not performing', 'poor performance', 'reduced output',
            'quality issue', 'product quality', 'process problem'
        }
        
        # Setup sentence transformers if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logging.info("Sentence transformer model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load sentence transformer model: {e}")
                self.embedding_model = None
        
        # Setup SpaCy if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(self.spacy_model)
                logging.info(f"SpaCy model '{self.spacy_model}' loaded successfully")
            except OSError:
                logging.warning(f"SpaCy model '{self.spacy_model}' not found. Install with: python -m spacy download {self.spacy_model}")
                self.nlp = None
            except Exception as e:
                logging.error(f"Failed to load SpaCy model: {e}")
                self.nlp = None
    
    def _load_cache(self) -> Dict:
        """Load classification cache from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                logging.info(f"Loaded cache with {len(cache)} entries")
                return cache
        except Exception as e:
            logging.error(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save classification cache to file."""
        try:
            # Convert numpy types to native Python types for JSON serialization
            cache_copy = {}
            for key, value in self.cache.items():
                if isinstance(value, dict):
                    cache_copy[key] = {}
                    for k, v in value.items():
                        if isinstance(v, (np.integer, np.floating)):
                            cache_copy[key][k] = v.item()
                        elif isinstance(v, list):
                            cache_copy[key][k] = [item.item() if isinstance(item, (np.integer, np.floating)) else item for item in v]
                        else:
                            cache_copy[key][k] = v
                else:
                    cache_copy[key] = value
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_copy, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save cache: {e}")
    
    def load_failure_dictionary(self, dict_path: str) -> bool:
        """Load failure mode dictionary from Excel file."""
        try:
            dict_df = pd.read_excel(dict_path)
            if not all(col in dict_df.columns for col in ['Keyword', 'Code', 'Description']):
                logging.error("Dictionary must contain Keyword, Code, and Description columns")
                return False
            
            # Build failure codes dictionary
            self.failure_codes = {}
            for _, row in dict_df.iterrows():
                code = str(row['Code'])
                description = str(row['Description'])
                keywords = str(row['Keyword'])
                
                if code not in self.failure_codes:
                    self.failure_codes[code] = {
                        'description': description,
                        'keywords': keywords,
                        'examples': []
                    }
                else:
                    # Handle multiple keywords for same code
                    self.failure_codes[code]['keywords'] += f", {keywords}"
            
            # Generate embeddings for failure codes if available
            if self.embedding_model and self.failure_codes:
                self._generate_failure_embeddings()
            
            logging.info(f"Loaded {len(self.failure_codes)} failure codes")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load failure dictionary: {e}")
            return False
    
    def analyze_historical_patterns(self, work_orders_df: pd.DataFrame):
        """Analyze historical patterns for temporal analysis"""
        if work_orders_df is not None and not work_orders_df.empty:
            self.temporal_analysis.analyze_temporal_patterns(work_orders_df)
            logging.info("Historical patterns analyzed for temporal classification")
    
    def _generate_failure_embeddings(self):
        """Generate embeddings for all failure codes"""
        try:
            failure_texts = []
            failure_codes_list = []
            
            for code, data in self.failure_codes.items():
                # Combine description and keywords for embedding
                text = f"{data['description']} {data['keywords']}"
                failure_texts.append(text)
                failure_codes_list.append(code)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(failure_texts, convert_to_tensor=True)
            
            # Store embeddings
            for i, code in enumerate(failure_codes_list):
                self.failure_embeddings[code] = embeddings[i]
            
            logging.info(f"Generated embeddings for {len(self.failure_codes)} failure codes")
            
        except Exception as e:
            logging.error(f"Failed to generate failure embeddings: {e}")
    
    def analyze_with_spacy(self, description: str) -> Dict:
        """Analyze work order description using SpaCy NLP."""
        if not self.nlp:
            return {}
        
        try:
            doc = self.nlp(description.lower())
            
            # Extract named entities
            entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'CARDINAL']]
            
            # Extract equipment types
            equipment_found = []
            for token in doc:
                if token.text in self.equipment_types:
                    equipment_found.append(token.text)
                # Also check noun phrases
                elif token.pos_ == 'NOUN' and any(equip in token.text for equip in self.equipment_types):
                    equipment_found.append(token.text)
            
            # Extract failure indicators
            failure_terms = []
            for token in doc:
                if token.text in self.failure_indicators:
                    failure_terms.append(token.text)
                # Check for failure-related adjectives
                elif token.pos_ == 'ADJ' and any(indicator in token.text for indicator in self.failure_indicators):
                    failure_terms.append(token.text)
            
            # Extract technical terms (nouns and adjectives)
            technical_terms = [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 3]
            
            # Extract verb phrases related to failures
            failure_verbs = [token.text for token in doc if token.pos_ == 'VERB' and any(indicator in token.text for indicator in self.failure_indicators)]
            
            return {
                'entities': entities,
                'equipment_types': list(set(equipment_found)),
                'failure_indicators': list(set(failure_terms)),
                'technical_terms': list(set(technical_terms)),
                'failure_verbs': list(set(failure_verbs)),
                'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
                'pos_tags': [(token.text, token.pos_) for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze with SpaCy: {e}")
            return {}
    
    def classify_with_spacy(self, description: str) -> Optional[AIClassificationResult]:
        """Classify failure mode using SpaCy NLP analysis."""
        if not self.nlp or not self.failure_codes:
            return None
        
        # Check cache first
        cache_key = f"spacy_{hash(description)}"
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            return AIClassificationResult(**cached_result)
        
        try:
            # Analyze with SpaCy
            analysis = self.analyze_with_spacy(description)
            
            if not analysis:
                return None
            
            # Score each failure code based on SpaCy analysis
            scores = {}
            for code, data in self.failure_codes.items():
                score = 0.0
                keywords = data['keywords'].lower().split(',')
                description_lower = description.lower()
                
                # Score based on keyword matches
                for keyword in keywords:
                    keyword = keyword.strip()
                    if keyword in description_lower:
                        score += 0.3
                    
                    # Check if keyword appears in technical terms
                    if keyword in analysis.get('technical_terms', []):
                        score += 0.2
                    
                    # Check if keyword appears in noun chunks
                    if any(keyword in chunk for chunk in analysis.get('noun_chunks', [])):
                        score += 0.15
                
                # Score based on equipment type matches
                for equip_type in analysis.get('equipment_types', []):
                    if any(equip_type in keyword for keyword in keywords):
                        score += 0.25
                
                # Score based on failure indicators
                for indicator in analysis.get('failure_indicators', []):
                    if any(indicator in keyword for keyword in keywords):
                        score += 0.2
                
                # Score based on entity matches
                for entity in analysis.get('entities', []):
                    if any(entity.lower() in keyword for keyword in keywords):
                        score += 0.1
                
                # Normalize score to 0-1 range
                scores[code] = min(1.0, score)
            
            # Find best match
            if scores:
                best_code = max(scores, key=scores.get)
                best_score = scores[best_code]
                
                # Use lower threshold for SpaCy since it's more conservative
                spacy_threshold = min(self.confidence_threshold, 0.3)
                
                if best_score >= spacy_threshold:
                    # Cache the result
                    result = AIClassificationResult(
                        code=best_code,
                        description=self.failure_codes[best_code]['description'],
                        confidence=best_score,
                        method='ai_spacy',
                        matched_keyword='',
                        reasoning=f"SpaCy analysis score: {best_score:.3f}",
                        entities=analysis.get('entities', []),
                        equipment_type=', '.join(analysis.get('equipment_types', [])),
                        failure_indicators=analysis.get('failure_indicators', [])
                    )
                    
                    self.cache[cache_key] = {
                        'code': result.code,
                        'description': result.description,
                        'confidence': result.confidence,
                        'method': result.method,
                        'matched_keyword': result.matched_keyword,
                        'reasoning': result.reasoning,
                        'entities': result.entities,
                        'equipment_type': result.equipment_type,
                        'failure_indicators': result.failure_indicators
                    }
                    self._save_cache()
                    
                    return result
        
        except Exception as e:
            logging.error(f"Failed to classify with SpaCy: {e}")
        
        return None

    def classify_with_openai(self, description: str) -> Optional[AIClassificationResult]:
        """Classify failure mode using OpenAI GPT"""
        # This method is no longer used as OpenAI is removed.
        # Keeping it for now to avoid breaking existing calls, but it will always return None.
        logging.warning("OpenAI classification is no longer available.")
        return None
    
    def classify_with_embeddings(self, description: str) -> Optional[AIClassificationResult]:
        """Classify failure mode using sentence embeddings"""
        if not self.embedding_model or not self.failure_embeddings:
            return None
        
        # Check cache first
        cache_key = f"embeddings_{hash(description)}"
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            return AIClassificationResult(**cached_result)
        
        try:
            # Generate embedding for the description
            if self.embedding_model is None:
                return None
            desc_embedding = self.embedding_model.encode([description], convert_to_tensor=True)
            
            # Calculate similarities with all failure codes
            similarities = {}
            for code, embedding in self.failure_embeddings.items():
                similarity = cosine_similarity(
                    desc_embedding.cpu().numpy().reshape(1, -1),
                    embedding.cpu().numpy().reshape(1, -1)
                )[0][0]
                similarities[code] = similarity
            
            # Find best match
            if similarities:
                best_code = max(similarities, key=similarities.get)
                best_similarity = similarities[best_code]
                
                # Convert similarity to confidence (0-1 scale)
                # Normalize similarity scores better for embeddings
                confidence = max(0.0, min(1.0, (best_similarity + 1) / 2))  # Convert from [-1,1] to [0,1]
                
                # Lower threshold for embeddings - they tend to have lower absolute scores
                embedding_threshold = min(self.confidence_threshold, 0.4)
                
                # If confidence is too low, use default code
                if confidence < embedding_threshold:
                    best_code = "0.0"
                    confidence = 0.1
                
                # Cache the result
                result = AIClassificationResult(
                    code=best_code,
                    description=self.failure_codes.get(best_code, {}).get('description', 'No Failure Mode Identified'),
                    confidence=confidence,
                    method='ai_embeddings',
                    matched_keyword='',
                    reasoning=f"Embedding similarity: {best_similarity:.3f}, normalized confidence: {confidence:.3f}"
                )
                
                self.cache[cache_key] = {
                    'code': result.code,
                    'description': result.description,
                    'confidence': result.confidence,
                    'method': result.method,
                    'matched_keyword': result.matched_keyword,
                    'reasoning': result.reasoning,
                    'entities': result.entities,
                    'equipment_type': result.equipment_type,
                    'failure_indicators': result.failure_indicators
                }
                self._save_cache()
                
                return result
        
        except Exception as e:
            logging.error(f"Failed to classify with embeddings: {e}")
        
        return None
    
    def classify_hybrid(self, description: str, dictionary_fallback_func) -> AIClassificationResult:
        """Hybrid classification: try AI methods first, fallback to dictionary"""
        
        # Try Expert System (rule-based)
        expert_code, expert_desc, expert_confidence = self.expert_system.classify(description)
        if expert_confidence >= self.confidence_threshold:
            return AIClassificationResult(
                code=expert_code,
                description=expert_desc,
                confidence=expert_confidence,
                method='expert_system',
                matched_keyword='',
                reasoning=f"Expert System score: {expert_confidence:.3f}"
            )
        
        # Try Contextual Pattern Recognition (equipment-specific)
        equipment_context_code, equipment_context_desc, equipment_context_confidence, equipment_type = self.contextual_patterns.classify_with_context(description)
        if equipment_context_confidence >= self.confidence_threshold:
            return AIClassificationResult(
                code=equipment_context_code,
                description=equipment_context_desc,
                confidence=equipment_context_confidence,
                method='contextual_patterns',
                matched_keyword='',
                reasoning=f"Contextual Pattern score: {equipment_context_confidence:.3f}",
                equipment_type=equipment_type
            )
        
        # Try Temporal Analysis (historical patterns)
        # Assuming current_date is available or can be passed
        # For now, using a placeholder or assuming it's not needed for temporal analysis
        # If work_orders_df is available, pass it to TimeSeriesPatternClassifier
        # For now, temporal analysis will always return '0.0' if no history
        temporal_code, temporal_desc, temporal_confidence = self.temporal_analysis.classify_with_temporal_context(description, 'unknown', datetime.now()) # Placeholder for current_date
        if temporal_confidence >= self.confidence_threshold:
            return AIClassificationResult(
                code=temporal_code,
                description=temporal_desc,
                confidence=temporal_confidence,
                method='temporal_analysis',
                matched_keyword='',
                reasoning=f"Temporal Pattern score: {temporal_confidence:.3f}"
            )
        
        # Try SpaCy NLP analysis (if available)
        if self.nlp:
            ai_result = self.classify_with_spacy(description)
            if ai_result and ai_result.confidence >= self.confidence_threshold:
                return ai_result
        
        # Try embeddings (always available if model is loaded)
        if self.embedding_model:
            ai_result = self.classify_with_embeddings(description)
            if ai_result:
                # Use embeddings result even with lower confidence for local processing
                embedding_threshold = min(self.confidence_threshold, 0.4)
                if ai_result.confidence >= embedding_threshold:
                    return ai_result
                # If confidence is low but not zero, still use it with lower confidence
                elif ai_result.confidence > 0.1:
                    ai_result.confidence = max(0.2, ai_result.confidence * 0.8)  # Boost confidence slightly
                    return ai_result
        
        # Fallback to dictionary matching
        try:
            code, desc, keyword = dictionary_fallback_func(description)
            return AIClassificationResult(
                code=code,
                description=desc,
                confidence=0.5,  # Medium confidence for dictionary fallback
                method='dictionary_fallback',
                matched_keyword=keyword,
                reasoning="AI confidence below threshold, using dictionary matching"
            )
        except Exception as e:
            logging.error(f"Dictionary fallback failed: {e}")
            return AIClassificationResult(
                code="0.0",
                description="No Failure Mode Identified",
                confidence=0.0,
                method='dictionary_fallback',
                matched_keyword='',
                reasoning="All classification methods failed"
            )
    
    def batch_classify(self, descriptions: List[str], dictionary_fallback_func) -> List[AIClassificationResult]:
        """Classify multiple descriptions efficiently"""
        results = []
        
        for i, description in enumerate(descriptions):
            try:
                result = self.classify_hybrid(description, dictionary_fallback_func)
                results.append(result)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logging.info(f"Processed {i + 1}/{len(descriptions)} descriptions")
                
            except Exception as e:
                logging.error(f"Failed to classify description {i}: {e}")
                results.append(AIClassificationResult(
                    code="0.0",
                    description="No Failure Mode Identified",
                    confidence=0.0,
                    method='error',
                    matched_keyword='',
                    reasoning=f"Classification error: {str(e)}"
                ))
        
        return results
    
    def get_classification_stats(self) -> Dict:
        """Get statistics about classification performance"""
        stats = {
            'total_classifications': len(self.cache),
            'methods_used': {},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'cache_size_mb': os.path.getsize(self.cache_file) / (1024 * 1024) if os.path.exists(self.cache_file) else 0
        }
        
        for cached_result in self.cache.values():
            method = cached_result.get('method', 'unknown')
            stats['methods_used'][method] = stats['methods_used'].get(method, 0) + 1
            
            confidence = cached_result.get('confidence', 0)
            if confidence >= 0.8:
                stats['confidence_distribution']['high'] += 1
            elif confidence >= 0.5:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
        
        return stats
    
    def export_training_data(self, output_file: str, work_orders_df: pd.DataFrame):
        """Export classification data for training/analysis"""
        try:
            training_data = []
            
            for _, row in work_orders_df.iterrows():
                description = str(row.get('Description', ''))
                failure_code = str(row.get('Failure Code', ''))
                failure_desc = str(row.get('Failure Description', ''))
                
                if description and failure_code:
                    # Add SpaCy analysis if available
                    spacy_analysis = {}
                    if self.nlp:
                        analysis = self.analyze_with_spacy(description)
                        spacy_analysis = {
                            'entities': analysis.get('entities', []),
                            'equipment_types': analysis.get('equipment_types', []),
                            'failure_indicators': analysis.get('failure_indicators', []),
                            'technical_terms': analysis.get('technical_terms', [])
                        }
                    
                    training_data.append({
                        'description': description,
                        'assigned_code': failure_code,
                        'assigned_description': failure_desc,
                        'spacy_analysis': spacy_analysis,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'work_order_analysis'
                    })
            
            # Save to JSON
            with open(output_file, 'w') as f:
                json.dump(training_data, f, indent=2)
            
            logging.info(f"Exported {len(training_data)} training examples to {output_file}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to export training data: {e}")
            return False
    
    def clear_cache(self):
        """Clear the classification cache"""
        self.cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        logging.info("Classification cache cleared") 