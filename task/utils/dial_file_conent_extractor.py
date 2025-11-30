import io
from pathlib import Path

import pandas as pd
import pdfplumber
from aidial_client import Dial
from aidial_client.types.file import FileDownloadResponse
from bs4 import BeautifulSoup


class DialFileContentExtractor:
    def __init__(self, endpoint: str, api_key: str):
        self.dial_client = Dial(base_url=endpoint, api_key=api_key)

    def extract_text(self, file_url: str) -> str:
        file: FileDownloadResponse = self.dial_client.files.download(file_url)

        file_name = file.filename
        file_extension = Path(file_name).suffix.lower()

        content = file.get_content()

        return self.__extract_text(
            file_content=content, file_extension=file_extension, filename=file_name
        )

    def __extract_text(
        self, file_content: bytes, file_extension: str, filename: str
    ) -> str:
        """Extract text content based on file type."""
        try:
            if file_extension == ".txt":
                return file_content.decode("utf-8", errors="ignore")

            if file_extension == ".pdf":
                pdf_bytes = io.BytesIO(file_content)
                pdf = pdfplumber.open(pdf_bytes)

                content = []
                for page in pdf.pages:
                    content.append(page.extract_text())

                return "\n".join(content)

            if file_extension == ".csv":
                buffer = io.StringIO(file_content.decode("utf-8", errors="ignore"))
                data_frame = pd.read_csv(buffer)
                markdown = data_frame.to_markdown(index=False)
                return markdown or ""

            if file_extension in [".html", ".htm"]:
                soup = BeautifulSoup(
                    markup=file_content.decode("utf-8", errors="ignore"),
                    features="html.parser",
                )

                for script in soup(["script", "style"]):
                    script.decompose()

                return soup.get_text(separator="\n", strip=True)

            return file_content.decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"Error while parsing {filename}: {e}")
            return ""
