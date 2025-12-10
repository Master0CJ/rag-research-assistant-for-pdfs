import logging
import os
import tempfile

from config import Config
from flask import Flask, jsonify, render_template, request
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.vector_store import VectorStore

# from services.llm_service import LLMService
from services.Q_bot import EducationalLLMService
from services.researchpanalysis import ResearchPaperAnalyzer
from services.storage_service import S3Storage

# add code for donwload .env from gdrive - refer chatgpt


app = Flask(__name__)
vector_store = VectorStore(Config.VECTOR_DB_PATH)
storage_service = S3Storage()
# llm_service = LLMService(vector_store)
Q_bot = EducationalLLMService(
    vector_store
)  # NEW: Initialize the educational LLM service
research_analyzer = ResearchPaperAnalyzer(
    Q_bot, vector_store
)  # Initialize ResearchPaperAnalyzer


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/summarizer")
def summarizer():
    return render_template("summarizer.html")


@app.route("/methodology")
def methodology():
    return render_template("methodology.html")


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def process_document(file):
    """Process document based on file type and return text chunks"""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        # Save file temporarily
        file.save(temp_path)

        # Process based on file type
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        elif file.filename.endswith(".txt"):
            loader = TextLoader(temp_path)
            documents = loader.load()
        else:
            raise ValueError("Unsupported file type")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        text_chunks = text_splitter.split_documents(documents)

        return text_chunks

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        os.rmdir(temp_dir)


@app.route("/upload", methods=["POST"])
def upload_document():
    try:
        logger.debug("Upload endpoint called")

        if "file" not in request.files:
            logger.warning("No file in request")
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            logger.warning("Empty filename")
            return jsonify({"error": "No file selected"}), 400

        # Check file extension
        if not file.filename.endswith((".txt", ".pdf")):
            logger.warning(f"Unsupported file type: {file.filename}")
            return jsonify({"error": "Only .txt and .pdf files are supported"}), 400

        logger.debug(f"Processing file: {file.filename}")

        # Process the document
        try:
            text_chunks = process_document(file)
            logger.debug(f"Document processed into {len(text_chunks)} chunks")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return jsonify({"error": f"Error processing document: {str(e)}"}), 500

        # Upload to S3
        try:
            file.seek(0)  # Reset file pointer
            storage_service.upload_file(file, file.filename)
            logger.debug("File uploaded to S3")
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            return jsonify({"error": f"Error uploading to S3: {str(e)}"}), 500

        # Add to vector store
        try:
            vector_store.add_documents(text_chunks)
            logger.debug("Documents added to vector store")
        except Exception as e:
            logger.error(f"Error adding to vector store: {str(e)}")
            return jsonify({"error": f"Error adding to vector store: {str(e)}"}), 500

        return jsonify(
            {
                "message": "File uploaded and processed successfully",
                "chunks_processed": len(text_chunks),
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    if "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    try:
        response = Q_bot.get_course_answer(data["question"])
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Research Paper Summarizer Routes
@app.route("/summarizer/upload", methods=["POST"])
def summarizer_upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported"}), 400

        # Upload to S3 first
        file.seek(0)
        storage_service.upload_file(file, file.filename)

        return jsonify(
            {
                "message": "Research paper uploaded successfully",
                "filename": file.filename,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/summarizer/summarize", methods=["POST"])
def summarizer_summarize():
    try:
        data = request.json
        filename = data.get("filename")
        title = data.get("title", "Research Paper")
        authors = data.get("authors", "Unknown")
        journal = data.get("journal", "Unknown")
        year = data.get("year", "Unknown")

        # Create paper metadata
        paper_metadata = {
            "title": title,
            "authors": authors,
            "journal": journal,
            "year": year,
        }

        # Download file from S3 to temporary location
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, filename)

        try:
            # Download from S3
            if not storage_service.get_file(
                object_name=filename, filename=temp_file_path
            ):
                return jsonify({"error": "Failed to retrieve file from storage"}), 500

            # Extract text from PDF
            pdf_text = research_analyzer._extract_pdf_text(temp_file_path)

            # Generate summary with the extracted text
            result = research_analyzer._generate_paper_summary(pdf_text, paper_metadata)

            return jsonify(
                {
                    "summary": result["answer"]
                    if isinstance(result, dict) and "answer" in result
                    else str(result),
                    "filename": filename,
                    "metadata": {
                        "title": title,
                        "authors": authors,
                        "journal": journal,
                        "year": year,
                    },
                }
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            os.rmdir(temp_dir)

    except Exception as e:
        logger.error(f"Error in summarizer_summarize: {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Research Methodology Extraction Routes
@app.route("/methodology/upload", methods=["POST"])
def methodology_upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported"}), 400

        # Upload to S3 first
        file.seek(0)
        storage_service.upload_file(file, file.filename)

        return jsonify(
            {
                "message": "Research paper uploaded successfully",
                "filename": file.filename,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/methodology/extract", methods=["POST"])
def methodology_extract():
    try:
        data = request.json
        filename = data.get("filename")

        # Use the research analyzer to extract methodology
        result = research_analyzer.extract_methodology(filename)

        return jsonify(
            {
                "methodology": result["answer"]
                if isinstance(result, dict) and "answer" in result
                else str(result),
                "filename": filename,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
