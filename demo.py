import os  
import logging  
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.output_parsers import StrOutputParser  
from langchain_openai import AzureChatOpenAI  
# from langchain_ollama.llms import OllamaLLM  
from langchain_text_splitters import CharacterTextSplitter  
from multiprocessing import Pool, current_process, Manager  
from functools import partial  
from criticalvectors import CriticalVectors  
from dotenv import load_dotenv  
  
# Optional MongoDB import  
try:  
    from pymongo import MongoClient  
except ImportError:  
    MongoClient = None  # Handle cases where pymongo is not installed  
  
# Global configuration for Azure OpenAI  
AZURE_API_VERSION = "2024-10-21"  
AZURE_DEPLOYMENT = "gpt-4o-mini"  
load_dotenv()  
  
# Configure logging  
logging.basicConfig(level=logging.INFO)  # Set level to DEBUG for more verbose output  
logger = logging.getLogger(__name__)  
  
def init_critical_vectors():  
    """  
    Initialize the critical vectors selector.  
    """  
    return CriticalVectors(  
        strategy='kmeans',  
        num_clusters='auto',  
        chunk_size=10000,  
        chunks_per_cluster=1,  # Set the desired number of chunks per cluster here  
        split_method='sentences',  
        max_tokens_per_chunk=3000,  # Adjust as needed  
        use_faiss=True  # Enable FAISS if desired  
    )  
  
def initialize_llm():  
    """  
    Initialize the preferred LLM instance.  
    This function is called within each worker process.  
    """  
    return AzureChatOpenAI(  # or OllamaLLM or whatever you prefer  
        openai_api_version=AZURE_API_VERSION,  
        azure_deployment=AZURE_DEPLOYMENT,  
        azure_endpoint=os.getenv("OPENAI_API_BASE"),  
        api_key=os.getenv("OPENAI_API_KEY")  
    )  
  
def worker_map_func(args):  
    """  
    Worker function to generate a summary for a single document.  
    This function is picklable and initializes its own LLM instance.  
    Returns a tuple of (summary, debug_info).  
    """  
    doc_content, map_prompt_template, process_name, logs_list = args  
    try:  
        # Initialize LLM within the worker  
        llm = initialize_llm()  
  
        # Create the map chain  
        map_chain = map_prompt_template | llm | StrOutputParser()  
  
        # Invoke the chain with the document content  
        summary = map_chain.invoke({"context": doc_content})  
  
        # Collect debug info  
        debug_info = {  
            'doc_content': doc_content,  
            'summary': summary  
        }  
  
        message = f"Process {process_name} summarized a chunk."  
        logger.debug(message)  
        logs_list.append(('DEBUG', message))  
  
        return summary, debug_info  
    except Exception as e:  
        error_msg = f"Error in process {process_name}: {e}"  
        logger.error(error_msg)  
        # Return empty summary and error info  
        debug_info = {  
            'doc_content': doc_content,  
            'error': str(e)  
        }  
        logs_list.append(('ERROR', error_msg))  
        return "", debug_info  
  
class Run:  
    """  
    Class to track all data related to a summarization run.  
    """  
    def __init__(self, strategy, parallelization):  
        self.strategy = strategy  
        self.parallelization = parallelization  
        self.original_text = ""  
        self.chunks = []  
        self.chunk_summaries = []  
        self.final_summary = ""  
        self.debug_info = {  
            'map_phase': [],  
            'reduce_phase': None,  
            'errors': []  
        }  
        self.logs = []
        self.prompt_metadata = {}
  
    def log(self, level, message):  
        # Log the message using the logging module  
        if level == 'DEBUG':  
            logger.debug(message)  
        elif level == 'INFO':  
            logger.info(message)  
        elif level == 'WARNING':  
            logger.warning(message)  
        elif level == 'ERROR':  
            logger.error(message)  
        elif level == 'CRITICAL':  
            logger.critical(message)  
        else:  
            logger.info(message)  
        # Store the log message  
        self.logs.append((level, message))  
  
    def to_dict(self):  
        """  
        Serialize the Run object to a dictionary for MongoDB storage.  
        """  
        return {  
            'strategy': self.strategy,  
            'parallelization': self.parallelization,  
            'original_text': self.original_text,  
            'chunks': self.chunks,  
            'chunk_summaries': self.chunk_summaries,  
            'final_summary': self.final_summary,  
            'debug_info': self.debug_info,  
            'logs': self.logs,
            'prompt_metadata': self.prompt_metadata  
        }  
  
class TextSummAI:  
    def __init__(  
        self,  
        strategy='map_reduce',  
        parallelization=True,  
        mongodb_uri=None,  
        mongodb_db_name=None,  
        mongodb_collection_name=None  
    ):  
        self.allowed_strategies = ['map_reduce','none','critical_vectors']  
        if strategy not in self.allowed_strategies:  
            raise Exception(f"Strategy {strategy} is not allowed. Allowed strategies: {self.allowed_strategies}")  
        self.strategy = strategy  
        if not isinstance(parallelization, bool):  
            raise ValueError("parallelization must be a boolean value")  
        self.parallelization = parallelization  
  
        # MongoDB support  
        self.mongodb_enabled = False  
        self.mongodb_client = None  
        self.mongodb_collection = None  
  
        if mongodb_uri and mongodb_db_name and mongodb_collection_name:  
            if MongoClient is None:  
                raise ImportError("pymongo is not installed. Please install it to use MongoDB support.")  
            try:  
                self.mongodb_client = MongoClient(mongodb_uri)  
                self.mongodb_db = self.mongodb_client[mongodb_db_name]  
                self.mongodb_collection = self.mongodb_db[mongodb_collection_name]  
                self.mongodb_enabled = True  
                logger.info("MongoDB support enabled.")  
            except Exception as e:  
                logger.error(f"Failed to connect to MongoDB: {e}")  
                self.mongodb_enabled = False  
  
        # Define map and reduce prompts  
        self.map_prompt = ChatPromptTemplate.from_messages([  
            ("human", "Write a concise summary of the following:\n\n{context}")  
        ])  
  
        self.reduce_prompt = ChatPromptTemplate.from_messages([  
            ("human", "Combine these summaries into a final summary:\n\n{summaries}")  
        ])  
  
        # Initialize reduce chain (single process)  
        self.reduce_llm = initialize_llm()  
        self.reduce_chain = self.reduce_prompt | self.reduce_llm | StrOutputParser()  
  
        # Initialize critical vectors selector (single process)  
        if self.strategy == 'critical_vectors':  
            self.critical_vectors = init_critical_vectors()  
        else:  
            self.critical_vectors = None  
  
        logger.info(f"TextSummAI initialized with strategy: {self.strategy}, parallelization: {self.parallelization}")  
  
    def summarize_text(self, text, chunk_size=1000, chunk_overlap=0):  
        # Create a new Run instance to track this summarization run  
        run = Run(strategy=self.strategy, parallelization=self.parallelization)  
        run.original_text = text  
  
        run.log('INFO', "Starting text summarization")  
  
        if self.strategy == 'map_reduce':  
            # Split the text into chunks  
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(  
                chunk_size=chunk_size, chunk_overlap=chunk_overlap  
            )  
            docs = text_splitter.create_documents([text])  
            run.chunks = [doc.page_content for doc in docs]  
  
            run.log('DEBUG', f"Text split into {len(run.chunks)} chunks.")  
  
            # Prepare the map function arguments  
            manager = Manager()  
            logs_list = manager.list()  
            processes_info = [  
                (chunk, self.map_prompt, f"Worker-{i+1}", logs_list)  
                for i, chunk in enumerate(run.chunks)  
            ]  
  
            # Generate summaries using parallel or sequential processing  
            if self.parallelization:  
                run.log('INFO', "Starting parallel processing of chunks.")  
                with Pool() as pool:  
                    results = pool.map(worker_map_func, processes_info)  
            else:  
                run.log('INFO', "Starting sequential processing of chunks.")  
                results = [worker_map_func(args) for args in processes_info]  
  
            # Unpack summaries and collect debug info  
            summaries = []  
            for summary, debug_info in results:  
                summaries.append(summary)  
                run.chunk_summaries.append(summary)  
                run.debug_info['map_phase'].append(debug_info)  
                if 'error' in debug_info:  
                    run.debug_info['errors'].append(debug_info['error'])  
  
            # Collect logs from subprocesses  
            for log_entry in logs_list:  
                run.logs.append(log_entry)  
  
            run.log('INFO', "All chunks processed. Proceeding to reduction step.")  
  
            # Combine the summaries into a final summary  
            final_summary = self.reduce_chain.invoke({"summaries": "\n\n".join(summaries)})  
            run.final_summary = final_summary  
  
            # Optionally, store reduce phase debug info  
            run.debug_info['reduce_phase'] = {  
                'input_summaries': summaries,  
                'final_summary': final_summary  
            }  
  
            run.log('INFO', "Summarization complete.")  
  
            # Show prompt details
            run.log('INFO', "Prompt details:")  
            run.log('INFO', f"Map prompt: {str(self.map_prompt)}\nReduce prompt: {str(self.reduce_prompt)}")
            
            run.prompt_metadata = {
                'map_prompt': str(self.map_prompt),
                'reduce_prompt': str(self.reduce_prompt)
            }
            # Save the run to MongoDB if enabled  
            if self.mongodb_enabled:  
                try:  
                    self.mongodb_collection.insert_one(run.to_dict())  
                    run.log('INFO', "Run data saved to MongoDB.")  
                except Exception as e:  
                    error_msg = f"Failed to save run data to MongoDB: {e}"  
                    run.log('ERROR', error_msg)  
                    logger.error(error_msg)  
  
            return final_summary, run  
  
        elif self.strategy == 'none':  
            # Return the original text, summarized with no chunking  
            run.log('INFO', "Summarizing without chunking.")  
            # Prepare the map function with the map_prompt  
            args = (str(text), self.map_prompt, "MainProcess", [])  
            summary, debug_info = worker_map_func(args)  
            run.final_summary = summary  
            run.debug_info['map_phase'].append(debug_info)  
            if 'error' in debug_info:  
                run.debug_info['errors'].append(debug_info['error'])  
            run.log('INFO', "Summarization complete.")  
            # Show prompt details
            run.log('INFO', "Prompt details:")  
            run.log('INFO', f"Map prompt: {str(self.map_prompt)}\nReduce prompt: {str(self.reduce_prompt)}")
            
            run.prompt_metadata = {
                'map_prompt': str(self.map_prompt),
                'reduce_prompt': str(self.reduce_prompt)
            }
            # Save the run to MongoDB if enabled  
            if self.mongodb_enabled:  
                try:  
                    self.mongodb_collection.insert_one(run.to_dict())  
                    run.log('INFO', "Run data saved to MongoDB.")  
                except Exception as e:  
                    error_msg = f"Failed to save run data to MongoDB: {e}"  
                    run.log('ERROR', error_msg)  
                    logger.error(error_msg)  
  
            return summary, run  
  
        elif self.strategy == 'critical_vectors':  
            # Use the critical vectors selector  
            run.log('INFO', "Summarizing using critical vectors strategy.")  
            if self.critical_vectors is None:  
                error_msg = "CriticalVectors selector is not initialized."  
                run.log('ERROR', error_msg)  
                raise Exception(error_msg)  
            selected_chunks, first_part, last_part = self.critical_vectors.get_relevant_chunks(str(text))  
            # Store chunks in run  
            run.chunks = selected_chunks  
            # Combine the parts to form the context  
            context = f"""  
[first part]  
{first_part}  
[/first part]  
  
[context]  
{' '.join(selected_chunks)}  
[/context]  
  
[last part]  
{last_part}  
[/last part]  
"""  
            args = (context, self.map_prompt, "MainProcess", [])  
            summary, debug_info = worker_map_func(args)  
            run.final_summary = summary  
            run.debug_info['map_phase'].append(debug_info)  
            if 'error' in debug_info:  
                run.debug_info['errors'].append(debug_info['error'])  
            run.log('INFO', "Summarization complete.")  
            # Show prompt details
            run.log('INFO', "Prompt details:")  
            run.log('INFO', f"Map prompt: {str(self.map_prompt)}\nReduce prompt: {str(self.reduce_prompt)}")
            
            run.prompt_metadata = {
                'map_prompt': str(self.map_prompt),
                'reduce_prompt': str(self.reduce_prompt)
            }
            # Save the run to MongoDB if enabled  
            if self.mongodb_enabled:  
                try:  
                    self.mongodb_collection.insert_one(run.to_dict())  
                    run.log('INFO', "Run data saved to MongoDB.")  
                except Exception as e:  
                    error_msg = f"Failed to save run data to MongoDB: {e}"  
                    run.log('ERROR', error_msg)  
                    logger.error(error_msg)  
  
            return summary, run  
  
if __name__ == "__main__":  
    # Ensure that the multiprocessing code runs only when the script is executed directly  
  
    # MongoDB connection details (optional)  
    mongodb_uri = os.getenv("MONGODB_URI")
    mongodb_db_name = "text_summ_ai"  
    mongodb_collection_name = "summarization_runs"  
  
    # Initialize the summarizer with parallelization enabled and MongoDB support  
    summarizer = TextSummAI(  
        strategy='critical_vectors',  
        parallelization=True,  
        mongodb_uri=mongodb_uri,  
        mongodb_db_name=mongodb_db_name,  
        mongodb_collection_name=mongodb_collection_name  
    )  
  
    # Example text  
    long_text = """  
    Your long text goes here...  
    """  
  
    # Summarize the text  
    summary, run = summarizer.summarize_text(long_text)  
  
    # Print the final summary  
    print("Final Summary:")  
    print(summary)  
  
    # For debugging: inspect the Run object  
    print("\nDebug Information:")  
    print(f"Original text length: {len(run.original_text)} characters")  
    print(f"Number of chunks: {len(run.chunks)}")  
    print("Chunks:")  
    for idx, chunk in enumerate(run.chunks):  
        print(f"Chunk {idx+1}: {chunk[:100]}...")  # Print first 100 characters  
  
    print("\nChunk Summaries:")  
    for idx, chunk_summary in enumerate(run.chunk_summaries):  
        print(f"Summary {idx+1}: {chunk_summary}")  
  
    if run.debug_info['errors']:  
        print("\nErrors encountered during summarization:")  
        for error in run.debug_info['errors']:  
            print(error)  
  
    # Optionally, print the logs collected in the Run object  
    print("\nLogs:")  
    for level, message in run.logs:  
        print(f"[{level}] {message}")  
