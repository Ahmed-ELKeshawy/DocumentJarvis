from flask import Flask, request, render_template, jsonify
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from pdfminer.high_level import extract_text
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_files'  # Create this folder in your project root

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#my own code


def load_pdf(file_path):
    fileText = extract_text(file_path)
    documentFile = split_text(fileText,file_path)
    print(f"Added {len(documentFile)} chunks for file: ",documentFile[0].metadata['Path'])


    return 0



def split_text(text,filename,path ,chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documentObject = splitter.split_documents([Document(page_content=text,metadata={"Path": path,"fileName": filename})])
    print("Document and MetaData Added")
    return documentObject

def addDocToVec(store,data):
    store.add_documents(documents = data )
    store.persist()


#end of my own code

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    for file in files:
        # Get the file's name and path
        filename = file.filename
        file_path = os.path.abspath(filename)
        file_path = file_path.replace("\\", "/")
        load_pdf(file_path)
        # Debugging info about file
        print(f"Processing file: {file_path}") 
    return jsonify({"message": "Files uploaded successfully!"})

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['query']
    # Placeholder response, replace with actual processing logic
    answer = f"Answer for query '{user_query}' based on uploaded documents."
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
