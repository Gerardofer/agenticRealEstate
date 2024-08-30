import os
import fitz  # PyMuPDF

class FileLoader:
  def __init__(self, file_path: str, encoding: str = "utf-8"):
      self.file_path = file_path
      self.documents = []
      self.encoding = encoding

  def load(self):
    if os.path.isdir(self.file_path):
        self.load_directory()
    elif os.path.isfile(self.file_path):
        if self.file_path.endswith(".pdf"):
            self.load_pdf_file()
        else: 
          raise ValueError(
              "Provided file is not a .txt or .pdf file."
          )
    else:
        raise ValueError(
            "Provided path is neither a valid directory nor a .txt or .pdf file."
        )

  def load_pdf_file(self):
      doc = fitz.open(self.file_path)
      text = ''
      for page in doc:
          text += page.get_text()
      self.documents.append(text)
      doc.close()

  def load_document(self):
      self.load()
      return self.documents
  
  # def __len__(self):
  #     return len(self.documents)
