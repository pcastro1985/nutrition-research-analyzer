from crewai import Agent, Task, Crew, Process, LLM
from src.models import PaperAnalysis

def create_nutrition_crew(paper_sections: dict, model_name: str, base_url: str) -> Crew:

    """
    Creates and returns a Crew configured to analyze a specific paper.
    """
    
    # FIX: Use CrewAI's native LLM class instead of LangChain
    # The model name for Ollama must be prefixed with 'ollama/'
    llm = LLM(
        model=f"ollama/{model_name}",
        base_url=base_url,
        temperature=0.1
    )

    triage_text = "\n\n".join(filter(None, [
        paper_sections.get("abstract"),
        paper_sections.get("funding"),
        paper_sections.get("conflicts_of_interest")
    ]))

    methods_text = paper_sections.get("methods") or "METHODS NOT REPORTED"

    stats_text = "\n\n".join(filter(None, [
        paper_sections.get("methods"),
        paper_sections.get("results")
    ]))

    title_text = paper_sections.get("title") or "TITLE NOT FOUND"

    # --- Agents ---

    title_agent = Agent(
        role='Title Extractor',
        goal='Extract the official title of the paper from the text.',
        backstory="You are a meticulous research assistant who always finds the exact title of scientific papers.",
        llm=llm,
        verbose=True
    )
    
    triage_agent = Agent(
        role='Nutrition Metadata Specialist',
        goal='Identify study type, funding sources, and conflicts of interest.',
        backstory="You are an investigative journalist specializing in scientific integrity. You look for industry funding in nutrition studies.",
        llm=llm,
        verbose=True
    )

    methodologist = Agent(
        role='Experimental Design Critic',
        goal='Scrutinize the study methodology, control groups, and confounding variables.',
        backstory="You are a senior scientist who hates bad study designs. You look for 'straw man' comparisons and healthy user bias.",
        llm=llm,
        verbose=True
    )

    statistician = Agent(
        role='Statistical Auditor',
        goal='Analyze results for relative vs absolute risk, surrogate markers, and p-hacking.',
        backstory="You are a statistician who doesn't trust headlines. You look at the data tables to find the real effect size.",
        llm=llm,
        verbose=True
    )

    summarizer = Agent(
        role='Lead Principal Investigator',
        goal='Synthesize reports from other agents into a final structured analysis.',
        backstory="You are the lead researcher. You take the findings from your team and produce the final verdict.",
        llm=llm,
        verbose=True
    )

    # --- Tasks (Direct Extraction) ---

    title_task = Task(
        description=f"""
        Extract the official paper title from the text below.

        Rules:
        - The title appears near the beginning of the text, usually right before the authors' names
        - Output the title EXACTLY as written
        - Do not include authors, journal names, or affiliations
        - If the title cannot be confidently identified, respond with NOT FOUND

        Text:
        {title_text}

        """,
        expected_output="The exact title of the paper as a string.",
        agent=title_agent
    )

    triage_task = Task(
        description=f"""
        Analyze the following paper text to extract metadata. First, classify the study type accurately, then assess evidence level, and identify funding/conflicts.

        Step 1: Classify Study Type (be precise):
        - RCT (randomized controlled trial)
        - Cohort (prospective/retrospective observational)
        - Case-control
        - Cross-sectional
        - Meta-analysis or systematic review
        - Case series/report
        - Other (specify)

        Step 2: Assign Evidence Level based on study type and quality indicators:
        - High: Well-conducted RCT or meta-analysis of RCTs
        - Medium: Cohort/case-control with adjustments, or lower-quality RCTs
        - Low: Cross-sectional, case reports, or studies with major flaws

        Step 3: Extract Funding and Conflicts:
        - Quote exact sources (e.g., 'Funding: Supported by Coca-Cola Foundation')
        - Flag industry ties (food/beverage/pharma companies)
        - Note government/academic/non-profit funding as neutral/positive

        Paper Text:
        {triage_text}

        If information is not explicitly present in the provided text, respond with NOT REPORTED. Do not infer.
        """,
        expected_output="""Structured bullet points:
        - Study Type: [exact type]
        - Evidence Level: [High/Med/Low] with brief rationale
        - Funding Sources: [quoted text or 'None stated']
        - Conflicts of Interest: [quoted text or 'None declared']""",
        agent=triage_agent
    )

    methodology_task = Task(
        description=f"""
        Analyze the Methods section of the paper, tailoring critique to the study type identified in triage.

        Common checks for ALL study types:
        - Sample size and power calculation (adequate?)
        - Participant selection/recruitment (representative? selection bias?)
        - Exposure/intervention measurement (validated tools? recall bias in observational?)
        - Confounders identified/adjusted (age, sex, BMI, smoking, diet, exercise?)
        - Specific biases by type:
          * RCTs: Randomization/blinding adequate? Control group appropriate (placebo/isocaloric)? Dropouts handled? Intention-to-treat analysis?
          * Cohort/Observational: Loss to follow-up? Healthy user bias (self-selected healthy diets)? Residual confounding?
          * Cross-sectional: Temporal ambiguity (chicken-egg)?
          * Meta-analysis: Search comprehensive? Heterogeneity assessed (I^2)? Publication bias (funnel plot)?
        - Dosage realistic and sustained compliance?

        Paper Text:
        {methods_text}

        If information is not explicitly present in the provided text, respond with NOT REPORTED. Do not infer.
        """,
        expected_output="""Critique in bullets:
        - Strengths: [list 1-3]
        - Weaknesses/Biases: [detailed, with quotes/page refs if possible]
        - Overall design quality: [Strong/Fair/Weak]""",
        agent=methodologist,
        context=[triage_task]
    )

    stats_task = Task(
        description=f"""
        Audit Results/Discussion sections for statistical rigor, focusing on effect sizes and significance.

        Step 1: Report key results with context:
        - Main findings: Quote effect sizes, CIs, p-values
        - Relative vs Absolute risk (e.g., RR 1.2 [1.1-1.3] = 20% rel increase, but abs 2% vs 1.7%)
        - Surrogate (e.g., LDL) vs hard outcomes (CVD events, mortality)

        Step 2: Check for issues:
        - P-hacking/multiplicity (too many tests? Bonferroni?)
        - Statistical vs clinical significance (tiny effect meaningful?)
        - Subgroup analyses (pre-specified? powered?)
        - Forest plots/tables scrutinized for outliers/heterogeneity

        Paper Text:
        {stats_text}

        If information is not explicitly present in the provided text, respond with NOT REPORTED. Do not infer.
        """,
        expected_output="""Bulleted audit:
        - Key Results: [RR/OR/HR with CIs; rel/abs]
        - Endpoints: [Surrogate/Clinical/Mixed]
        - Stat Issues: [p-hacking, multiplicity, etc.]
        - Trust in numbers: [High/Med/Low] why""",
        agent=statistician,
        context=[methodology_task]
    )

    synthesis_task = Task(
        description="""
        Synthesize triage, methodology, and stats reports into final PaperAnalysis.


        Guidelines:
        - Base trust score on evidence level, biases, funding (industry-funded observational: deduct heavily)
        - Be critical: Nutrition studies often overhype weak associations
        - Output ONLY valid JSON matching PaperAnalysis pydantic schema
        - Fill all required fields accurately from prior tasks
        - If information is not explicitly present in the provided text, respond with NOT REPORTED. Do not infer.
        
        IMPORTANT:
        - You MUST output ALL fields defined in the PaperAnalysis schema
        - If information is missing or unclear, set the field to null
        - Do NOT omit fields
        - Do NOT guess or infer beyond the provided evidence
        - Boolean fields must be true, false, or null
        - Never output "NOT REPORTED" for boolean fields

        Mapping rules for has_conflict_of_interest:
        - If conflicts or industry funding are explicitly stated → true
        - If authors explicitly state "no conflicts of interest" → false
        - If no disclosure is present or information is missing → null
        Do NOT output strings like "NOT REPORTED" for boolean fields.
        """,
        expected_output="A JSON object matching the PaperAnalysis schema.",
        agent=summarizer,
        context=[title_task, triage_task, methodology_task, stats_task],
        output_pydantic=PaperAnalysis
    )

    return Crew(
        agents=[title_agent, triage_agent, methodologist, statistician, summarizer],
        tasks=[title_task, triage_task, methodology_task, stats_task, synthesis_task],
        process=Process.sequential,
        verbose=True
    )