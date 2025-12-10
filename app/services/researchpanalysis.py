from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ResearchPaperAnalyzer:
    def __init__(self, llm_service, vector_store):
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for research papers
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " "],
        )

    def process_research_paper(self, pdf_file, paper_metadata=None):
        """Extract and analyze research paper content"""
        try:
            # Extract text from PDF
            text = self._extract_pdf_text(pdf_file)

            # Create structured document with metadata
            doc_metadata = {
                "document_type": "research_paper",
                "title": paper_metadata.get("title", "Unknown"),
                "authors": paper_metadata.get("authors", "Unknown"),
                "year": paper_metadata.get("year", "Unknown"),
                "journal": paper_metadata.get("journal", "Unknown"),
            }

            # Split into sections (Abstract, Introduction, Methods, Results, etc.)
            sections = self._identify_sections(text)
            documents = []

            for section_name, section_text in sections.items():
                chunks = self.text_splitter.split_text(section_text)
                for i, chunk in enumerate(chunks):
                    section_metadata = doc_metadata.copy()
                    section_metadata.update({"section": section_name, "chunk_id": i})
                    documents.append(
                        Document(page_content=chunk, metadata=section_metadata)
                    )

            # Add to vector store
            self.vector_store.add_documents(documents)

            # Generate paper summary
            summary = self._generate_paper_summary(text, doc_metadata)

            return {
                "success": True,
                "paper_id": doc_metadata["title"],
                "sections_processed": list(sections.keys()),
                "summary": summary,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_methodology(self, paper_title):
        """Extract and explain research methodology"""
        methodology_prompt = f'''
        From the paper "{paper_title}", extract and explain:
        1. Research design and approach
        2. Data collection methods
        3. Sample size and characteristics
        4. Statistical analysis techniques used
        5. Limitations of the methodology
        6. How this methodology could be applied to similar research'''

        return self.llm_service.get_course_answer(methodology_prompt, "methodology")

    def _extract_pdf_text(self, pdf_file_path):
        """Extract text from PDF file using PyPDFLoader"""
        try:
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()
            # Combine all pages into a single text
            text = "\n\n".join([doc.page_content for doc in documents])
            return text
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")

    def _identify_sections(self, text):
        """Identify common academic paper sections"""
        sections = {
            "abstract": "",
            "introduction": "",
            "literature_review": "",
            "methodology": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
            "references": "",
        }

        # Simple section identification based on common headings
        text_lower = text.lower()
        section_markers = {
            "abstract": ["abstract", "summary"],
            "introduction": ["introduction", "background"],
            "literature_review": ["literature review", "related work", "previous work"],
            "methodology": ["methodology", "methods", "experimental setup", "approach"],
            "results": ["results", "findings", "experiments"],
            "discussion": ["discussion", "analysis"],
            "conclusion": ["conclusion", "conclusions", "summary"],
            "references": ["references", "bibliography"],
        }

        # Basic section extraction (simplified approach)
        for section_name, markers in section_markers.items():
            for marker in markers:
                if marker in text_lower:
                    # Find section text (simplified - would need more robust parsing)
                    start_idx = text_lower.find(marker)
                    if start_idx != -1:
                        # Take next 500 characters as section content
                        sections[section_name] = text[start_idx : start_idx + 500]
                        break

        return sections

    def _generate_paper_summary(self, text, metadata):
        """Generate structured summary of research paper"""
        # Use a larger chunk of text for better summary quality (up to 8000 chars)
        # If text is longer, we'll use the first 8000 chars which should cover abstract, intro, and key sections
        text_preview = text[:8000] if len(text) > 8000 else text
        text_indicator = "..." if len(text) > 8000 else ""

        if not text or len(text.strip()) == 0:
            raise ValueError(
                "No text content provided for summarization. Please ensure the PDF file contains extractable text."
            )

        summary_prompt = f"""Summarize this research paper comprehensively:

Title: {metadata.get("title", "Unknown")}
Authors: {metadata.get("authors", "Unknown")}
Journal: {metadata.get("journal", "Unknown")}
Year: {metadata.get("year", "Unknown")}

Paper Content:
{text_preview}{text_indicator}

Please provide a detailed structured summary with the following sections:
1. Research Objective/Question: What is the main research question or objective?
2. Key Methodology: Describe the research approach, methods, and techniques used
3. Main Findings: Summarize the key results and findings
4. Implications: What are the practical or theoretical implications of these findings?
5. Limitations: What limitations or constraints are mentioned in the paper?
6. Conclusion: Provide a brief overall conclusion

Ensure your summary is comprehensive, accurate, and well-structured."""

        return self.llm_service.get_course_answer(summary_prompt, "summary")
