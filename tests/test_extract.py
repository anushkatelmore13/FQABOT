import unittest
import os
import tempfile
from src.extract import extract_text_from_file


class TestExtract(unittest.TestCase):
    def test_extract_txt_file(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
            f.write("Hello world. This is a test document for text extraction.")
            f_path = f.name

        try:
            result = extract_text_from_file(f_path, "test.txt")
            self.assertIn("Hello world", result)
            self.assertIn("text extraction", result)
        finally:
            if os.path.exists(f_path):
                os.remove(f_path)

    def test_extract_empty_txt(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
            f.write("   \n  ")
            f_path = f.name

        try:
            with self.assertRaises(ValueError):
                extract_text_from_file(f_path, "empty.txt")
        finally:
            if os.path.exists(f_path):
                os.remove(f_path)

    def test_unsupported_format(self):
        with self.assertRaises(ValueError):
            extract_text_from_file("sample.docx", "sample.docx")

    def test_extract_markdown_file(self):
        content = b"# Leave Policy\n\nEmployees receive 20 PTO days per year."
        result = extract_text_from_file(content, "policy.md")

        self.assertIn("Leave Policy", result)
        self.assertIn("20 PTO days", result)

    def test_extract_csv_file(self):
        content = b"question,answer\nRemote stipend?,Employees receive $1000 annually.\n"
        result = extract_text_from_file(content, "faq.csv")

        self.assertIn("Row 1", result)
        self.assertIn("question: Remote stipend?", result)
        self.assertIn("answer: Employees receive $1000 annually.", result)


if __name__ == "__main__":
    unittest.main()
