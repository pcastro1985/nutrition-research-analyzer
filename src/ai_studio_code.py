def generate_docx_report(results: list) -> BytesIO:
    """
    Generates a Word document from the list of analysis results.
    Returns a BytesIO object containing the .docx file.
    """
    doc = Document()
    
    # --- Title Page ---
    title = doc.add_heading('Nutrition Science Analysis Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"Total Papers Analyzed: {len(results)}")
    doc.add_page_break()

    # --- Loop through each paper ---
    for i, res in enumerate(results):
        # SAFER title logic - check title first, fallback to index/fallback
        if hasattr(res, 'title') and res.title and res.title.strip():
            paper_title = res.title[:100]  # Truncate long titles
        else:
            paper_title = f"Paper {i+1} ({get_filename_fallback(res)})"
            
        header = doc.add_heading(paper_title, level=1)
        
        # Sub-header details using exact schema fields
        p = doc.add_paragraph()
        p.add_run(f"Type: {getattr(res, 'paper_type', 'N/A')} | ").bold = True
        p.add_run(f"Evidence Level: {getattr(res, 'evidence_level', 'N/A')} | ").bold = True
        
        # Safe trust score access
        trust_score = getattr(res, 'trust_score', 0)
        run = p.add_run(f"Trust Score: {trust_score}/10")
        run.bold = True
        if isinstance(trust_score, (int, float)) and trust_score < 5:
            run.font.color.rgb = RGBColor(255, 0, 0)
        
        # 1. Conflicts of Interest
        doc.add_heading('💰 Conflicts of Interest & Funding', level=2)
        has_coi = getattr(res, 'has_conflict_of_interest', False)
        if has_coi:
            p = doc.add_paragraph()
            warning = p.add_run("⚠️ WARNING: CONFLICT DETECTED")
            warning.font.color.rgb = RGBColor(255, 0, 0)
            warning.bold = True
            doc.add_paragraph(f"Source: {getattr(res, 'funding_source', 'N/A')}")
            doc.add_paragraph(f"Notes: {getattr(res, 'coi_notes', 'N/A')}")
        else:
            doc.add_paragraph("No obvious conflicts of interest declared.")

        # 2. Methodology
        doc.add_heading('🔬 Methodology Audit', level=2)
        doc.add_paragraph(f"Control Group: {getattr(res, 'control_group_quality', 'N/A')}", style='List Bullet')
        doc.add_paragraph(f"Intervention: {getattr(res, 'intervention_details', 'N/A')}", style='List Bullet')
        doc.add_paragraph(f"Confounders: {getattr(res, 'confounding_factors', 'N/A')}", style='List Bullet')

        # 3. Statistics
        doc.add_heading('📈 Statistical Analysis', level=2)
        doc.add_paragraph(f"Primary Outcome: {getattr(res, 'primary_outcome', 'N/A')}", style='List Bullet')
        doc.add_paragraph(f"Risk Reported: {getattr(res, 'risk_type_reported', 'N/A')}", style='List Bullet')
        doc.add_paragraph(f"Endpoints: {getattr(res, 'endpoints', 'N/A')}", style='List Bullet')
        doc.add_paragraph(f"Significance: {getattr(res, 'statistical_significance', 'N/A')}", style='List Bullet')

        # 4. Conclusions
        doc.add_heading('🏁 Conclusions & Verdict', level=2)
        doc.add_paragraph(f"Authors' Conclusion: {getattr(res, 'conclusion_summary', 'N/A')}")
        doc.add_paragraph(getattr(res, 'final_verdict', 'N/A'), style='List Bullet')
        
        # Separator between papers
        doc.add_page_break()

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def get_filename_fallback(res) -> str:
    """Safe filename fallback - your app.py likely passes this."""
    # Common patterns where filename might be stored
    try:
        # Option 1: Check if results have filename from process_zip_file
        if hasattr(res, 'filename') and res.filename:
            return res.filename.replace('.pdf', '').title()
        # Option 2: Use generic fallback
        return "Untitled Analysis"
    except:
        return "Untitled Analysis"
