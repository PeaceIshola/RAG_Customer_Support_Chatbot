import glob
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader

def load_documents():
    """Load PDF files and web pages into raw documents."""
    pdf_paths = glob.glob("data/Everstorm_*.pdf")
    raw_docs = []

    for pdf_path in pdf_paths:
        raw_docs.extend(PyPDFLoader(pdf_path).load())

    print(f"Loaded {len(raw_docs)} PDF pages from {len(pdf_paths)} files.")

    URLS = [
        # --- BigCommerce – shipping & refunds ---
        "https://developer.bigcommerce.com/docs/store-operations/shipping",
        "https://developer.bigcommerce.com/docs/store-operations/orders/refunds",
        # --- Stripe – disputes & chargebacks ---
        # "https://docs.stripe.com/disputes",  
        # --- WooCommerce – REST API reference ---
        # "https://woocommerce.github.io/woocommerce-rest-api-docs/v3.html",
    ]

    try:
        loader = UnstructuredURLLoader(urls=URLS)
        raw_docs.extend(loader.load())
        print(f"Fetched {len(raw_docs)} documents from the web.")
    except Exception as e:
        print("⚠️  Web fetch failed, using offline copies:", e)
        raw_docs = []
        for pdf_path in pdf_paths:
            raw_docs.extend(PyPDFLoader(pdf_path).load())
        print(f"Loaded {len(raw_docs)} offline documents.")
    
    return raw_docs
