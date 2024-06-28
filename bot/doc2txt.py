# Script for converting .docx to .txt
import os
import sys
import docx2txt


def find_docx(docs_path):
    """
    Iterate over .docx files in every subdirectory
    """
    if not docs_path:
        raise ValueError("docs_path is required")

    # Iterate over .docx files in every subdirectory
    for root, dirs, files in os.walk(docs_path):
        for file in files:
            if file.endswith(".docx"):
                doc_path = os.path.join(root, file)
                if os.path.isfile(doc_path):
                    yield doc_path
                else:
                    print(f"Ignoring {doc_path} as it is not a file")

def doc2txt(docs_path, txt_path):
    if not docs_path:
        raise ValueError("docs_path is required")

    for doc_path in find_docx(docs_path):
        print(f"Converting {doc_path} to {txt_path}")

        if not os.path.exists(doc_path):
            print(f"Ignoring {doc_path} as it is not a file")
            continue

        txt_file = os.path.join(txt_path, os.path.basename(doc_path) + ".txt")

        if not os.path.exists(txt_path):
            os.makedirs(txt_path)

        if os.path.exists(txt_file):
            print(f"{txt_file} already exists. Skipping conversion.")
            continue

        try:
            doc = docx2txt.process(doc_path)
            with open(txt_file, "w") as f:
                f.write(doc)
        except Exception as e:
            print(f"Failed to convert {doc_path}: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python doc2txt.py <docs_path> <txt_path>")
        sys.exit(1)

    docs_path = sys.argv[1]
    txt_path = sys.argv[2]
    # find_docx(docs_path)
    # doc2txt(find_docx(docs_path), txt_path)
    doc2txt(docs_path, txt_path)