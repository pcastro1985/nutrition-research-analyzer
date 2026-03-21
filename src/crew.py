from crewai import Agent, Task, Crew, Process, LLM
from src.models import PaperAnalysis


def create_nutrition_crew(paper_sections: dict, model_config: dict) -> Crew:
    """
    Creates and returns a Crew configured to analyze a specific paper.
    model_config: dict with keys 'provider' ('ollama' or 'groq'), 'name', and optionally 'api_key', 'base_url'
    """

    provider = model_config.get("provider", "ollama")

    if provider == "groq":
        llm = LLM(
            model=model_config["name"],
            api_key=model_config.get("api_key"),
            base_url="https://api.groq.com/openai/v1",
            temperature=0.0,
        )
    else:
        llm = LLM(
            model=f"ollama/{model_config['name']}",
            base_url=model_config.get("base_url", "http://localhost:11434"),
            temperature=0.0,
        )

    triage_text = "\n\n".join(
        filter(
            None,
            [
                paper_sections.get("abstract"),
                paper_sections.get("funding"),
                paper_sections.get("conflicts_of_interest"),
            ],
        )
    )

    methods_text = paper_sections.get("methods") or "METHODS NOT REPORTED"

    stats_text = "\n\n".join(
        filter(None, [paper_sections.get("methods"), paper_sections.get("results")])
    )

    title_text = paper_sections.get("title") or "TITLE NOT FOUND"

    # Context for Classification (Abstract + Methods start)
    classification_text = "\n\n".join(
        filter(
            None,
            [
                f"TITLE: {paper_sections.get('title')}",
                f"ABSTRACT: {paper_sections.get('abstract')}",
                f"METHODS START: {(paper_sections.get('methods') or '')[:2000] if paper_sections.get('methods') else 'Methods missing'}",
            ],
        )
    )

    # Full text for summary
    summary_text = "\n\n".join(
        filter(
            None,
            [
                f"TITLE: {paper_sections.get('title')}",
                f"ABSTRACT: {paper_sections.get('abstract')}",
                f"METHODS: {paper_sections.get('methods') or 'Not available'}",
                f"RESULTS: {paper_sections.get('results') or 'Not available'}",
                f"CONCLUSIONS: {paper_sections.get('conclusion') or paper_sections.get('conclusions') or 'Not available'}",
            ],
        )
    )

    # --- Agents ---

    title_agent = Agent(
        role="Title Extractor",
        goal="Extract the official title of the paper from the text.",
        backstory="You are a meticulous research assistant who always finds the exact title of scientific papers.",
        llm=llm,
        verbose=True,
    )

    classifier_agent = Agent(
        role="Study Taxonomist",
        goal="Classify the scientific study design with 100% precision.",
        backstory="You are an expert taxonomist. You distinguish between RCTs, Cohorts, Metaanalysis, Reviews, etc. instantly. You never hallucinate study types.",
        llm=llm,
        verbose=True,
    )

    triage_agent = Agent(
        role="Nutrition Metadata Specialist",
        goal="Assign evidence level based on the Taxonomist report and find conflicts of interest.",
        backstory="You are an investigative journalist. You trust the Taxonomist's classification, and you hunt for industry funding.",
        llm=llm,
        verbose=True,
    )

    methodologist = Agent(
        role="Experimental Design Critic",
        goal="Scrutinize the study methodology for the specific design identified.",
        backstory="You are a senior scientist. You look for 'straw man' comparisons and healthy user bias.",
        llm=llm,
        verbose=True,
    )

    statistician = Agent(
        role="Statistical Auditor",
        goal="Analyze results for relative vs absolute risk, surrogate markers, and p-hacking.",
        backstory="You are a statistician who doesn't trust headlines. You look at the data tables to find the real effect size.",
        llm=llm,
        verbose=True,
    )

    summarizer = Agent(
        role="Lead Principal Investigator",
        goal="Synthesize reports from other agents into a final structured analysis.",
        backstory="You are the lead researcher. You take the findings from your team and produce the final verdict.",
        llm=llm,
        verbose=True,
    )

    summary_agent = Agent(
        role="Paper Summarizer",
        goal="Create a clear summary of the paper's objective, methodology, and conclusions.",
        backstory="You are a skilled scientific writer. You distill complex research into concise, readable summaries.",
        llm=llm,
        verbose=True,
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
        agent=title_agent,
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
        agent=classifier_agent,
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
        context=[classification_task],  # Takes input from classifier
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
        context=[classification_task],
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
        context=[methodology_task],
    )

    summary_task = Task(
        description=f"""
        Create a structured summary of the paper with three main sections:

        1. OBJECTIVE: What was the research question or aim? What was being studied?

        2. METHODOLOGY: How was the study conducted? Include:
           - Study design
           - Population/samples
           - Intervention/exposure (if applicable)
           - Key measurements

        3. CONCLUSIONS: What were the main findings and authors' interpretations?

        Text:
        {summary_text}

        If information is not explicitly present in the provided text, respond with NOT REPORTED. Do not infer.
        """,
        expected_output="""Structured summary with:
        - Objective: [2-3 sentences]
        - Methodology: [3-5 sentences]
        - Conclusions: [2-4 sentences]""",
        agent=summary_agent,
    )

    synthesis_task = Task(
        description="""
        Create the final JSON report.
        
        IMPORTANT INSTRUCTIONS FOR LLM:
        1. You are receiving inputs from Title, Taxonomist, Triage, Methodologist, Statistician, and Paper Summarizer.
        2. Combine these into the JSON format.
        3. **CRITICAL:** Do NOT output the JSON Schema definition (do not output "type": "string", "description": "...", etc).
        4. Output the ACTUAL DATA values extracted from the study.
        5. Use the Paper Summarizer's output to fill in the objective, methodology_summary, and conclusions fields.
        
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
        context=[
            title_task,
            classification_task,
            triage_task,
            methodology_task,
            stats_task,
            summary_task,
        ],
        output_pydantic=PaperAnalysis,
    )

    return Crew(
        agents=[
            title_agent,
            classifier_agent,
            triage_agent,
            methodologist,
            statistician,
            summary_agent,
            summarizer,
        ],
        tasks=[
            title_task,
            classification_task,
            triage_task,
            methodology_task,
            stats_task,
            summary_task,
            synthesis_task,
        ],
        process=Process.sequential,
        verbose=True,
    )
