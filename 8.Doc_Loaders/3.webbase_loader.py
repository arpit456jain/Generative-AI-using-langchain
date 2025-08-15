from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)


model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/nrt-360-degree-orange-single-rotating-trolley-wheel-appliance-furniture-caster/p/itmc650faa71a299?pid=FCAFZFKUQGJAGA44&lid=LSTFCAFZFKUQGJAGA44MDBPXR&marketplace=FLIPKART&fm=neo%2Fmerchandising&iid=M_2b64987f-358b-438e-8657-c815c45ccd18_4_ZV5UVKUJY6J7_MC.FCAFZFKUQGJAGA44&ppt=hp&ppn=homepage&ssid=1kwb619gnk0000001755189306769&otracker=clp_pmu_v2_Tools%2Band%2BAccessories_3_4.productCard.PMU_V2_NRT%2B360%2BDEGREE%2BORANGE%2BSINGLE%2BROTATING%2BTROLLEY%2BWHEEL%2BAppliance%2BFurniture%2BCaster_independence-day-sale-store_FCAFZFKUQGJAGA44_neo%2Fmerchandising_2&otracker1=clp_pmu_v2_PINNED_neo%2Fmerchandising_Tools%2Band%2BAccessories_LIST_productCard_cc_3_NA_view-all&cid=FCAFZFKUQGJAGA44'
loader = WebBaseLoader(url)

docs = loader.load()


chain = prompt | model | parser

print(chain.invoke({'question':'What is the prodcut that we are talking about?', 'text':docs[0].page_content}))