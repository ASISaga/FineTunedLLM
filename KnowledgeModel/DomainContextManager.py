"""
Domain Context Management System
This module provides domain-specific context management for focused learning
in both adaptive summarization and fine-tuning stages.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class DomainType(Enum):
    """Enumeration of supported domain types"""
    TECHNICAL = "technical"
    MEDICAL = "medical" 
    LEGAL = "legal"
    FINANCIAL = "financial"
    ACADEMIC = "academic"
    BUSINESS = "business"
    SCIENTIFIC = "scientific"
    GENERAL = "general"

@dataclass
class DomainContext:
    """Domain-specific context configuration"""
    domain_type: DomainType
    domain_name: str
    description: str
    key_concepts: List[str]
    terminology: Dict[str, str]
    focus_areas: List[str]
    summarization_instructions: str
    fine_tuning_instructions: str
    example_prompts: List[str]
    evaluation_criteria: List[str]

class DomainContextManager:
    """Manages domain-specific context for focused learning"""
    
    def __init__(self):
        self.domains: Dict[str, DomainContext] = {}
        self._load_default_domains()
    
    def _load_default_domains(self):
        """Load default domain configurations"""
        
        # Technical Domain
        technical_domain = DomainContext(
            domain_type=DomainType.TECHNICAL,
            domain_name="Software Development",
            description="Software engineering, programming, and technology documentation",
            key_concepts=[
                "algorithms", "data structures", "design patterns", "architecture",
                "frameworks", "APIs", "databases", "testing", "deployment"
            ],
            terminology={
                "API": "Application Programming Interface",
                "CI/CD": "Continuous Integration/Continuous Deployment",
                "ORM": "Object-Relational Mapping",
                "REST": "Representational State Transfer",
                "GraphQL": "Graph Query Language"
            },
            focus_areas=[
                "code quality", "performance optimization", "security best practices",
                "scalability", "maintainability", "documentation"
            ],
            summarization_instructions="""
            Focus on extracting technical concepts, implementation details, best practices,
            and actionable insights. Emphasize code patterns, architectural decisions,
            and performance considerations. Include specific technical terms and methodologies.
            """,
            fine_tuning_instructions="""
            Generate training examples that emphasize technical problem-solving,
            code analysis, architectural design, and best practices. Include
            diverse programming scenarios and technical decision-making contexts.
            """,
            example_prompts=[
                "Explain the implementation of a distributed caching system",
                "What are the trade-offs between microservices and monolithic architecture?",
                "How do you optimize database queries for large-scale applications?"
            ],
            evaluation_criteria=[
                "technical accuracy", "implementation feasibility", "best practices adherence",
                "performance considerations", "security implications"
            ]
        )
        
        # Medical Domain
        medical_domain = DomainContext(
            domain_type=DomainType.MEDICAL,
            domain_name="Healthcare and Medicine",
            description="Medical research, clinical practice, and healthcare documentation",
            key_concepts=[
                "diagnosis", "treatment", "pathology", "pharmacology", "anatomy",
                "clinical trials", "evidence-based medicine", "patient care"
            ],
            terminology={
                "EHR": "Electronic Health Record",
                "CPR": "Cardiopulmonary Resuscitation",
                "MRI": "Magnetic Resonance Imaging",
                "CT": "Computed Tomography",
                "ICU": "Intensive Care Unit"
            },
            focus_areas=[
                "patient safety", "clinical guidelines", "treatment protocols",
                "diagnostic accuracy", "medication management", "preventive care"
            ],
            summarization_instructions="""
            Extract key medical concepts, treatment protocols, diagnostic criteria,
            and clinical evidence. Focus on patient outcomes, safety considerations,
            and evidence-based recommendations. Maintain medical accuracy and clarity.
            """,
            fine_tuning_instructions="""
            Create training examples that emphasize clinical reasoning, medical
            knowledge application, and patient-centered care. Include diverse
            medical scenarios and diagnostic challenges.
            """,
            example_prompts=[
                "What are the differential diagnoses for chest pain in elderly patients?",
                "Explain the mechanism of action of ACE inhibitors",
                "Describe the latest treatment guidelines for diabetes management"
            ],
            evaluation_criteria=[
                "medical accuracy", "clinical relevance", "evidence-based reasoning",
                "patient safety", "guideline compliance"
            ]
        )
        
        # Legal Domain
        legal_domain = DomainContext(
            domain_type=DomainType.LEGAL,
            domain_name="Legal and Regulatory",
            description="Legal documents, regulations, and jurisprudence",
            key_concepts=[
                "contracts", "litigation", "compliance", "regulations", "statutes",
                "case law", "legal precedent", "due process", "jurisdiction"
            ],
            terminology={
                "GDPR": "General Data Protection Regulation",
                "SOX": "Sarbanes-Oxley Act",
                "HIPAA": "Health Insurance Portability and Accountability Act",
                "SEC": "Securities and Exchange Commission",
                "USPTO": "United States Patent and Trademark Office"
            },
            focus_areas=[
                "regulatory compliance", "risk assessment", "legal precedent",
                "contract analysis", "intellectual property", "data privacy"
            ],
            summarization_instructions="""
            Focus on legal principles, regulatory requirements, compliance obligations,
            and legal precedents. Extract key legal concepts, obligations, and
            potential risks. Maintain legal accuracy and cite relevant authorities.
            """,
            fine_tuning_instructions="""
            Generate training examples that emphasize legal reasoning, regulatory
            analysis, and compliance assessment. Include diverse legal scenarios
            and jurisdictional considerations.
            """,
            example_prompts=[
                "What are the key compliance requirements under GDPR?",
                "Analyze the enforceability of non-compete clauses",
                "Explain the legal implications of AI bias in hiring decisions"
            ],
            evaluation_criteria=[
                "legal accuracy", "regulatory compliance", "precedent analysis",
                "risk assessment", "jurisdictional awareness"
            ]
        )
        
        # Financial Domain
        financial_domain = DomainContext(
            domain_type=DomainType.FINANCIAL,
            domain_name="Finance and Investment",
            description="Financial analysis, investment strategies, and market research",
            key_concepts=[
                "portfolio management", "risk assessment", "financial modeling",
                "market analysis", "derivatives", "asset allocation", "valuation"
            ],
            terminology={
                "ROI": "Return on Investment",
                "NPV": "Net Present Value",
                "CAPM": "Capital Asset Pricing Model",
                "VaR": "Value at Risk",
                "ESG": "Environmental, Social, and Governance"
            },
            focus_areas=[
                "investment analysis", "risk management", "financial planning",
                "market trends", "regulatory compliance", "performance metrics"
            ],
            summarization_instructions="""
            Extract financial metrics, investment strategies, risk factors, and
            market insights. Focus on quantitative analysis, performance indicators,
            and financial implications. Include relevant financial calculations.
            """,
            fine_tuning_instructions="""
            Create training examples that emphasize financial analysis, investment
            decision-making, and risk assessment. Include diverse market scenarios
            and financial instruments.
            """,
            example_prompts=[
                "Analyze the risk-return profile of emerging market bonds",
                "What factors should be considered in ESG investing?",
                "Explain the impact of interest rate changes on bond portfolios"
            ],
            evaluation_criteria=[
                "financial accuracy", "quantitative rigor", "risk assessment",
                "market relevance", "regulatory awareness"
            ]
        )
        
        # Register default domains
        self.domains["technical"] = technical_domain
        self.domains["medical"] = medical_domain
        self.domains["legal"] = legal_domain
        self.domains["financial"] = financial_domain
    
    def get_domain_context(self, domain_name: str) -> Optional[DomainContext]:
        """Retrieve domain context by name"""
        return self.domains.get(domain_name.lower())
    
    def add_custom_domain(self, domain_context: DomainContext):
        """Add a custom domain context"""
        self.domains[domain_context.domain_name.lower()] = domain_context
        logger.info(f"Added custom domain: {domain_context.domain_name}")
    
    def list_available_domains(self) -> List[str]:
        """List all available domain names"""
        return list(self.domains.keys())
    
    def get_summarization_prompt(self, domain_name: str, base_prompt: str) -> str:
        """Generate domain-specific summarization prompt"""
        domain = self.get_domain_context(domain_name)
        if not domain:
            return base_prompt
        
        enhanced_prompt = f"""
        {base_prompt}
        
        DOMAIN CONTEXT: {domain.domain_name}
        {domain.description}
        
        KEY CONCEPTS TO FOCUS ON:
        {', '.join(domain.key_concepts)}
        
        FOCUS AREAS:
        {', '.join(domain.focus_areas)}
        
        DOMAIN-SPECIFIC INSTRUCTIONS:
        {domain.summarization_instructions}
        
        TERMINOLOGY AWARENESS:
        {json.dumps(domain.terminology, indent=2)}
        
        Please ensure your response incorporates domain expertise and uses appropriate terminology.
        """
        
        return enhanced_prompt
    
    def get_fine_tuning_context(self, domain_name: str) -> Dict[str, Any]:
        """Get domain-specific fine-tuning configuration"""
        domain = self.get_domain_context(domain_name)
        if not domain:
            return {}
        
        return {
            "domain_type": domain.domain_type.value,
            "domain_name": domain.domain_name,
            "key_concepts": domain.key_concepts,
            "focus_areas": domain.focus_areas,
            "fine_tuning_instructions": domain.fine_tuning_instructions,
            "example_prompts": domain.example_prompts,
            "evaluation_criteria": domain.evaluation_criteria,
            "terminology": domain.terminology
        }
    
    def generate_training_prompt_template(self, domain_name: str) -> str:
        """Generate domain-specific training prompt template"""
        domain = self.get_domain_context(domain_name)
        if not domain:
            return "Generate a helpful response to the following question: {question}"
        
        template = f"""
        You are an expert in {domain.domain_name}. {domain.description}
        
        Key areas of expertise include: {', '.join(domain.focus_areas)}
        
        When responding, please:
        1. Apply domain-specific knowledge and terminology
        2. Focus on {', '.join(domain.key_concepts[:3])} when relevant
        3. {domain.fine_tuning_instructions}
        
        Question: {{question}}
        
        Provide a comprehensive, domain-expert response:
        """
        
        return template
    
    def export_domain_config(self, domain_name: str) -> Optional[str]:
        """Export domain configuration as JSON"""
        domain = self.get_domain_context(domain_name)
        if not domain:
            return None
        
        config = {
            "domain_type": domain.domain_type.value,
            "domain_name": domain.domain_name,
            "description": domain.description,
            "key_concepts": domain.key_concepts,
            "terminology": domain.terminology,
            "focus_areas": domain.focus_areas,
            "summarization_instructions": domain.summarization_instructions,
            "fine_tuning_instructions": domain.fine_tuning_instructions,
            "example_prompts": domain.example_prompts,
            "evaluation_criteria": domain.evaluation_criteria
        }
        
        return json.dumps(config, indent=2)
    
    def import_domain_config(self, config_json: str) -> bool:
        """Import domain configuration from JSON"""
        try:
            config = json.loads(config_json)
            domain_context = DomainContext(
                domain_type=DomainType(config["domain_type"]),
                domain_name=config["domain_name"],
                description=config["description"],
                key_concepts=config["key_concepts"],
                terminology=config["terminology"],
                focus_areas=config["focus_areas"],
                summarization_instructions=config["summarization_instructions"],
                fine_tuning_instructions=config["fine_tuning_instructions"],
                example_prompts=config["example_prompts"],
                evaluation_criteria=config["evaluation_criteria"]
            )
            
            self.add_custom_domain(domain_context)
            return True
        except Exception as e:
            logger.error(f"Failed to import domain config: {str(e)}")
            return False
