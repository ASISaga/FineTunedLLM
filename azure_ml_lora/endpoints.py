"""
Azure ML endpoint configurations
"""
import os

# Agent endpoint mappings
AML_ENDPOINTS = {
    "cmo": {
        "scoring_uri": os.getenv("AML_CMO_SCORING_URI", ""),
        "key": os.getenv("AML_CMO_KEY", "")
    },
    "cfo": {
        "scoring_uri": os.getenv("AML_CFO_SCORING_URI", ""),
        "key": os.getenv("AML_CFO_KEY", "")
    },
    "cto": {
        "scoring_uri": os.getenv("AML_CTO_SCORING_URI", ""),
        "key": os.getenv("AML_CTO_KEY", "")
    }
}