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
        temperature=0.0
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

    # Context for Classification (Abstract + Methods start)
    classification_text = "\n\n".join(filter(None, [
        f"TITLE: {paper_sections.get('title')}",
        f"ABSTRACT: {paper_sections.get('abstract')}",
        f"METHODS START: {paper_sections.get('methods')[:2000] if paper_sections.get('methods') else 'Methods missing'}"
    ]))

    # --- Agents ---

    title_agent = Agent(
        role='Title Extractor',
        goal='Extract the official title of the paper from the text.',
        backstory="You are a meticulous research assistant who always finds the exact title of scientific papers.",
        llm=llm,
        verbose=True
    )
    
    classifier_agent = Agent(
        role='Study Taxonomist',
        goal='Classify the scientific study design with 100% precision.',
        backstory="You are an expert taxonomist. You distinguish between RCTs, Cohorts, Metaanalysis, Reviews, etc. instantly. You never hallucinate study types.",
        llm=llm,
        verbose=True
    )

    triage_agent = Agent(
        role='Nutrition Metadata Specialist',
        goal='Assign evidence level based on the Taxonomist report and find conflicts of interest.',
        backstory="You are an investigative journalist. You trust the Taxonomist's classification, and you hunt for industry funding.",
        llm=llm,
        verbose=True
    )

    methodologist = Agent(
        role='Experimental Design Critic',
        goal='Scrutinize the study methodology for the specific design identified.',
        backstory="You are a senior scientist. You look for 'straw man' comparisons and healthy user bias.",
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

    classification_task = Task(
        description=f"""
        Analyze the Abstract and Methods below to classify the study design.
        
        Options (Choose ONE):
        - RCT (Randomized Controlled Trial)
        - Cohort Study (Prospective/Retrospective)
        - Cross-Sectional Study
        - Case-Control Study
        - Systematic Review / Meta-Analysis
        - Narrative Review
        - Animal/In-vitro Study
        
        Look for keywords: "randomized", "double-blind" (RCT); "followed up", "baseline" (Cohort); "snapshot", "survey" (Cross-sectional).

        Text:
        {classification_text}
        """,
        expected_output="The exact study design type (e.g., 'RCT').",
        agent=classifier_agent
    )

    # UPDATED TASK: Triage (Depends on Classification)
    triage_task = Task(
        description=f"""
        1. Read the Study Classification provided by the 'Study Taxonomist'.
        2. Assign an Evidence Level (High/Medium/Low) based on that classification.
           - High: RCT, Meta-Analysis
           - Medium: Cohort, Case-Control
           - Low: Cross-sectional, Animal, Narrative Review
        3. Analyze the text below for Conflicts of Interest (COI) and Funding.
        
        Text:
        {triage_text}

        If information is not explicitly present in the provided text, respond with NOT REPORTED. Do not infer.
        """,
        expected_output="Evidence Level and COI details.",
        agent=triage_agent,
        context=[classification_task] # Takes input from classifier
    )

    methodology_task = Task(
        description=f"""
        Critique the Methods. Use the Study Type identified by the Taxonomist (in context) to guide your critique.
        
        Check for:
        - Control group quality
        - Dosage/Intervention realism
        - Confounding adjustments (if observational)
        - Randomization methods (if RCT)

        Text:
        {methods_text}

        If information is not explicitly present in the provided text, respond with NOT REPORTED. Do not infer.
        """,
        expected_output="Methodology strengths and weaknesses.",
        agent=methodologist,
        context=[classification_task] 
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
        Create the final JSON report.
        
        IMPORTANT INSTRUCTIONS FOR LLM:
        1. You are receiving inputs from Title, Taxonomist, Triage, Methodologist, and Statistician.
        2. Combine these into the JSON format.
        3. **CRITICAL:** Do NOT output the JSON Schema definition (do not output "type": "string", "description": "...", etc).
        4. Output the ACTUAL DATA values extracted from the study.
        
        Example of CORRECT output:
        {
            "title": "Study of X vs Y",
            "paper_type": "RCT",
            "trust_score": 8,
            ...
        }

        Example of WRONG output (DO NOT DO THIS):
        {
            "properties": { "title": { "type": "string" } }
        }
        
        If a field is unknown, use null.
        """,
        expected_output="A valid JSON object matching the PaperAnalysis schema with extracted data.",
        agent=summarizer,
        context=[title_task, classification_task, triage_task, methodology_task, stats_task],
        output_pydantic=PaperAnalysis
    )

    return Crew(
        agents=[title_agent, classifier_agent, triage_agent, methodologist, statistician, summarizer],
        tasks=[title_task, classification_task, triage_task, methodology_task, stats_task, synthesis_task],
        process=Process.sequential,
        verbose=True
    )