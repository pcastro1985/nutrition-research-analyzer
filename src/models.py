from pydantic import BaseModel, Field
from typing import Optional

class PaperAnalysis(BaseModel):
    """Structured output for the nutrition paper analysis."""
    filename: Optional[str] = Field(
        None, description="Original PDF filename")
    title: Optional[str] = Field(
        None, description="Extracted title")
    paper_type: Optional[str] = Field(
        None, description="Type of study (RCT, Systematic Review, Observational, etc.)")
    evidence_level: Optional[str] = Field(
        None, description="High, Medium, or Low based on hierarchy of evidence")
    
    # Conflict of Interest
    has_conflict_of_interest: Optional[bool] = Field(
        None, description="True if industry funding/COI detected, False if explicitly declared none, None if not reported")
    funding_source: Optional[str] = Field(
        None, description="Who funded the study?")
    coi_notes: Optional[str] = Field(
        None, description="Details on the conflict of interest")

    # Methodology
    control_group_quality: Optional[str] = Field(
        None, description="Analysis of the control group (Fair comparison vs Straw man)")
    intervention_details: Optional[str] = Field(
        None, description="Dosage and adherence details")
    confounding_factors: Optional[str] = Field(
        None, description="Potential confounding variables or Healthy User Bias")

    # Statistics
    primary_outcome: Optional[str] = Field(
        None, description="Primary outcome reported by authors, or None if not clearly stated")
    risk_type_reported: Optional[str] = Field(
        None, description="Type of risk reported (RR, OR, HR), or None if not reported")
    endpoints: Optional[str] = Field(
        None, description="Surrogate markers (e.g. Cholesterol) vs Clinical Endpoints (e.g. Heart Attack)")
    statistical_significance: Optional[str] = Field(
        None, description="Summary of statistical significance, or None if not reported")

    # Conclusions
    conclusion_summary: Optional[str] = Field(
        None, description="Authors' conclusions, or None if unclear")
    trust_score: Optional[int] = Field(
        None, ge=1, le=10, description="Model-derived trust score (1–10), None if insufficient evidence")
    final_verdict: Optional[str] = Field(
        None, description="Overall evidence verdict, or None if cannot be determined")