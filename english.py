import os, tiktoken
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index import ServiceContext, set_global_tokenizer,set_global_service_context
from llama_index import set_global_handler

# Set up logging
# set_global_handler("wandb", run_args={"project": "llamaindex"})


from llama_index import(
    VectorStoreIndex,
    KeywordTableIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    get_response_synthesizer,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    load_indices_from_storage
)

from llama_index import QueryBundle
from llama_index.callbacks.base import CallbackManager
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import NodeWithScore

#Retirever
from llama_index.retrievers import(
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

#Query Engine
from llama_index.query_engine import RetrieverQueryEngine

from typing import List, Optional

set_global_tokenizer(
    tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)
token_counter = TokenCountingHandler(verbose=False)
callback_manager = CallbackManager([token_counter])

#Load the documents
documents = SimpleDirectoryReader("files").load_data()

#Initialize the Service Context
service_context = ServiceContext.from_defaults(chunk_size=1500, chunk_overlap=200, callback_manager=callback_manager)
set_global_service_context(service_context=service_context)
node_parser = service_context.node_parser
nodes = node_parser.get_nodes_from_documents(documents)

token_counter.reset_counts()
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    #Initialize the Storage Context
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    storage_context.docstore.persist()

    #Build the Vector Index
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    vector_index.storage_context.persist()

    #Build Keyword Index
    keyword_index = KeywordTableIndex(nodes, storage_context=storage_context)
    vector_index.storage_context.persist()

else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    vector_index, keyword_index = load_indices_from_storage(storage_context)

#Building the Custom retriever
class CustomRetiever(BaseRetriever):
    #Custom Retriever that performs both semantic and keyword search

    def __init__(
        self,
        vector_retrirver : VectorIndexRetriever,
        keyword_retriever : KeywordTableSimpleRetriever,
        mode : str = "AND"
    ) -> None:
        self._vector_retrirver = vector_retrirver
        self._keyword_retriever = keyword_retriever

        if mode not in ("AND", "OR"):
            raise ValueError("Invalid Mode")
        
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        #Retrive nodes given a query
        vector_nodes = self._vector_retrirver.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        print("VECTOR IDS ----->", vector_ids)
        print("KEYWORD ID ----->", keyword_ids)

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retriever_ids = vector_ids.intersection(keyword_ids)
        else :
            retriever_ids = vector_ids.union(keyword_ids)

        retriever_nodes = [combined_dict[rid] for rid in retriever_ids]
        return retriever_nodes
    
#define custom retiver
vector_retrirver = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
custom_retiever = CustomRetiever(vector_retrirver, keyword_retriever, "OR")

#define the response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="refine")

from llama_index.prompts import PromptTemplate, MessageRole

text_qa_template = (
    "You are an excellent and a topper student. Below is the context to the question.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query in the style of a studious student who will score the best marks. Make sure to answer it pointwise staring with point a)\n"
    "Query: {query_str}\n"
    "Answer: "
)

refine_template = f'''
You are a highly professional and excellent English Teacher who cares a lot about how and what is taught to the stidents.
Given the question, explain it to the students in a very excellent manner. Keep that do not miss any detail and being
an English teacher the answer should be highlt accurate and professional.
'''
qa_prompt_tmpl = PromptTemplate(text_qa_template)
#assemble query engine
custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retiever,
    response_synthesizer=response_synthesizer
)
custom_query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)


#get the response
response = custom_query_engine.query("Describe how Sadao did the surgery? This is a 10 mark question, so give a very detailed answer")
print(response)
print("---------------------------------------------------------\n\n\n")

print("Emdebbing Token :", token_counter.total_embedding_token_count)
print("LLM Prompt Token:", token_counter.prompt_llm_token_count)
print("LLM Completion Token:", token_counter.completion_llm_token_count)
print("Total LLM Token Count:", token_counter.total_llm_token_count)

print("prompt: ", token_counter.llm_token_counts[0].prompt, "...\n\n\n\n")

print(
    "completion: ", token_counter.llm_token_counts[0].completion, "...\n"
)