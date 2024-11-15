import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from pdfminer.high_level import extract_text
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentQAInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Document Retrieval and Q&A Interface")
        self.root.geometry("800x800")

        # Initialize components before creating widgets
        self.initialize_components()
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('Header.TLabel', font=('Arial', 16, 'bold'))
        self.style.configure('Subheader.TLabel', font=('Arial', 14, 'bold'))
        
        # Create upload folder
        self.upload_folder = 'uploaded_files'
        os.makedirs(self.upload_folder, exist_ok=True)
        
        self.create_widgets()


    def initialize_components(self):
        """Initialize all required components for the QA system"""
        try:
            # Initialize status to track progress
            self.initialization_status = "Starting initialization...\n"
            
            # Initialize embeddings
            self.initialization_status += "Initializing embeddings...\n"
            self.embeddings = GPT4AllEmbeddings(client=any)
            
            # Initialize vector store
            self.initialization_status += "Initializing vector store...\n"
            self.vectorstore = Chroma(
                persist_directory="TesterFinal",
                embedding_function=self.embeddings,
                
            )
            
            # Initialize LLM
            self.initialization_status += "Initializing Ollama...\n"
            self.llm = Ollama(model="llama3.2:1b")
            
            # Initialize prompt template
            self.initialization_status += "Setting up prompt template...\n"
            self.prompt_template = """
            Context: {context}
            Question: {question}
            Answer the question based on the context provided. If you cannot find 
            the answer in the context, say "I cannot find the answer in the provided documents."
            Answer:
            """
            self.QA_PROMPT = PromptTemplate(
                template=self.prompt_template,
                input_variables=["context", "question"]
            )
            
            # Initialize QA chain
            self.initialization_status += "Setting up QA chain...\n"
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": self.QA_PROMPT},
                return_source_documents=True
            )
            
            self.initialization_status += "Initialization completed successfully!\n"
            
        except Exception as e:
            self.initialization_status = f"Error during initialization: {str(e)}\n"
            print(f"Initialization error: {str(e)}")

    def create_widgets(self):
        # Main container with padding
        main_container = ttk.Frame(self.root, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(
            main_container, 
            text="Document Retrieval and Q&A Interface",
            style='Header.TLabel'
        )
        title.pack(pady=(0, 20))
        
        # Upload Section
        upload_section = ttk.Frame(main_container)
        upload_section.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(
            upload_section,
            text="Upload Documents",
            style='Subheader.TLabel'
        ).pack(pady=(0, 10))
        
        upload_frame = ttk.Frame(upload_section)
        upload_frame.pack(fill=tk.X)
        
        self.file_list = ttk.Treeview(
            upload_frame,
            columns=('Path', 'Filename'),
            show='headings',
            height=5
        )
        self.file_list.heading('Path', text='File Path')
        self.file_list.heading('Filename', text='Filename')
        self.file_list.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        scrollbar = ttk.Scrollbar(upload_frame, orient=tk.VERTICAL, command=self.file_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_list.configure(yscrollcommand=scrollbar.set)
        
        button_frame = ttk.Frame(upload_section)
        button_frame.pack(pady=10)
        
        ttk.Button(
            button_frame,
            text="Select Files",
            command=self.select_files
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Upload Files",
            command=self.upload_files
        ).pack(side=tk.LEFT, padx=5)
        
        # Upload Result Label
        self.upload_result = ttk.Label(upload_section, text="", foreground="green")
        self.upload_result.pack(pady=5)
        
        # Query Section
        query_section = ttk.Frame(main_container)
        query_section.pack(fill=tk.X, pady=20)
        
        ttk.Label(
            query_section,
            text="Enter Your Query",
            style='Subheader.TLabel'
        ).pack(pady=(0, 10))
        
        self.query_entry = ttk.Entry(query_section, width=50)
        self.query_entry.pack(pady=(0, 10))
        
        ttk.Button(
            query_section,
            text="Submit Query",
            command=self.submit_query
        ).pack()
        
        # Query Result Area
        self.query_result = scrolledtext.ScrolledText(
            query_section,
            wrap=tk.WORD,
            width=60,
            height=10
        )
        self.query_result.pack(pady=10)
        
    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select PDF files",
            filetypes=[("PDF files", "*.pdf")]
        )
        
        # Clear existing items
        for item in self.file_list.get_children():
            self.file_list.delete(item)
            
        # Add new files to the list
        for file_path in files:
            filename = os.path.basename(file_path)
            self.file_list.insert('', tk.END, values=(file_path, filename))
    
    def upload_files(self):
        for item in self.file_list.get_children():
            file_path = self.file_list.item(item)['values'][0]
            filename = self.file_list.item(item)['values'][1]
            try:
                # Process the file using your existing functions
                self.load_pdf(file_path)
                self.upload_result.config(
                    text="Files uploaded successfully!",
                    foreground="green"
                )
            except Exception as e:
                self.upload_result.config(
                    text=f"Error uploading files: {str(e)}",
                    foreground="red"
                )
    
    def submit_query(self):
        query = self.query_entry.get()
        if query:
            try:
                # Use the QA chain to get the answer
                result = self.qa_chain({"query": query})
                
                # Get the answer and source documents
                answer = result['result']
                source_documents = result["source_documents"]
                
                # Extract unique file names
                unique_file_names = {doc.metadata.get("fileName", "Unknown File") 
                                for doc in source_documents}
                
                # Prepare the complete response text
                response_text = f"Answer:\n{answer}\n\n"
                response_text += "Context retrieved from the following document(s):\n"
                for file_name in unique_file_names:
                    response_text += f"- {file_name}\n"
                
                # Update the query result text area
                self.query_result.delete(1.0, tk.END)
                self.query_result.insert(tk.END, response_text)
                
                # Print to console as well for debugging
                print("Answer:", answer)
                print("\nContext retrieved from the following document(s):")
                for file_name in unique_file_names:
                    print("-", file_name)
                    
            except Exception as e:
                error_message = f"Error processing query: {str(e)}"
                self.query_result.delete(1.0, tk.END)
                self.query_result.insert(tk.END, error_message)
                print(error_message)
    
    def load_pdf(self, file_path):
        fileText = extract_text(file_path)
        filename = os.path.basename(file_path)
        documentFile = self.split_text(fileText, filename, file_path)
        print(f"Added {len(documentFile)} chunks for file: ", documentFile[0].metadata['Path'])
        self.addDocToVec(documentFile)

        return 
    

    def addDocToVec(self,data):
        self.vectorstore.add_documents(documents = data )
        self.vectorstore.persist()
        print('added files to vectore store\n')
        return

    def split_text(self, text, filename, path, chunk_size=1000, chunk_overlap=100):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        documentObject = splitter.split_documents([
            Document(
                page_content=text,
                metadata={"Path": path, "fileName": filename}
            )
        ])
        print("Document and MetaData Added")
        return documentObject
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DocumentQAInterface()
    app.run()