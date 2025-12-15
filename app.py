import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="বাংলা FAQ চ্যাটবট",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Setup Embeddings
@st.cache_resource
def load_embedding_model():
    """Load the Bengali embedding model"""
    return HuggingFaceEmbeddings(model_name="l3cube-pune/bengali-sentence-similarity-sbert")

# Prepare FAQ Dataset with Metadata
def prepare_faq_data():
    """Prepare FAQ chunks with metadata for 5 Bangla topics"""
    
    # শিক্ষা (Education)
    education_chunks = [
        ("বিশ্ববিদ্যালয়ে ভর্তির জন্য এসএসসি এবং এইচএসসিতে ভালো ফলাফল প্রয়োজন।", 
         {"category": "শিক্ষা", "difficulty": "সহজ"}),
        ("অনলাইন কোর্স করার জন্য ইন্টারনেট সংযোগ এবং একটি ডিভাইস লাগবে।", 
         {"category": "শিক্ষা", "difficulty":  "সহজ"}),
        ("স্কুলে ভর্তির জন্য জন্ম নিবন্ধন সনদ প্রয়োজন।", 
         {"category": "শিক্ষা", "difficulty": "সহজ"}),
        ("উচ্চশিক্ষার জন্য স্কলারশিপ পেতে ভালো একাডেমিক রেকর্ড থাকতে হবে।", 
         {"category": "শিক্ষা", "difficulty":  "মাঝারি"}),
        ("বিদেশে পড়াশোনার জন্য IELTS বা TOEFL স্কোর প্রয়োজন হতে পারে।", 
         {"category": "শিক্ষা", "difficulty": "মাঝারি"}),
    ]
    
    # স্বাস্থ্য (Health)
    health_chunks = [
        ("সর্দি-কাশির জন্য গরম পানি পান এবং বিশ্রাম নিন।", 
         {"category": "স্বাস্থ্য", "difficulty": "সহজ"}),
        ("প্রতিদিন কমপক্ষে ৮ গ্লাস পানি পান করা উচিত।", 
         {"category": "স্বাস্থ্য", "difficulty": "সহজ"}),
        ("জ্বর হলে প্যারাসিটামল খাওয়া যেতে পারে তবে ডাক্তারের পরামর্শ নিন।", 
         {"category": "স্বাস্থ্য", "difficulty": "সহজ"}),
        ("নিয়মিত ব্যায়াম শরীর সুস্থ রাখতে সাহায্য করে।", 
         {"category": "স্বাস্থ্য", "difficulty": "সহজ"}),
        ("রক্তচাপ নিয়ন্ত্রণে রাখতে লবণ কম খান এবং স্ট্রেস কমান।", 
         {"category": "স্বাস্থ্য", "difficulty":  "মাঝারি"}),
    ]
    
    # ভ্রমণ (Travel)
    travel_chunks = [
        ("কক্সবাজার ভ্রমণের জন্য প্রায় ১৫-২০ হাজার টাকা বাজেট রাখুন।", 
         {"category": "ভ্রমণ", "difficulty": "সহজ"}),
        ("সুন্দরবন যেতে হলে খুলনা বা সাতক্ষীরা থেকে বোটে যেতে হবে।", 
         {"category": "ভ্রমণ", "difficulty": "সহজ"}),
        ("সিলেটের জাফলং এবং রাতারগুল খুব সুন্দর পর্যটন স্থান।", 
         {"category": "ভ্রমণ", "difficulty": "সহজ"}),
        ("পাসপোর্ট করতে অনলাইনে আবেদন করে ফি জমা দিতে হয়।", 
         {"category": "ভ্রমণ", "difficulty": "মাঝারি"}),
        ("বিদেশ ভ্রমণের জন্য ভিসা প্রয়োজন হতে পারে, দেশভেদে ভিন্ন।", 
         {"category": "ভ্রমণ", "difficulty": "মাঝারি"}),
    ]
    
    # প্রযুক্তি (Technology)
    technology_chunks = [
        ("স্মার্টফোনের ব্যাটারি বাঁচাতে ব্রাইটনেস কমান এবং অপ্রয়োজনীয় অ্যাপ বন্ধ করুন।", 
         {"category": "প্রযুক্তি", "difficulty": "সহজ"}),
        ("ইন্টারনেট স্পিড বাড়াতে রাউটার রিস্টার্ট দিন এবং সঠিক স্থানে রাখুন।", 
         {"category": "প্রযুক্তি", "difficulty": "সহজ"}),
        ("কম্পিউটার স্লো হলে অপ্রয়োজনীয় ফাইল ডিলিট করুন এবং অ্যান্টিভাইরাস চালান।", 
         {"category": "প্রযুক্তি", "difficulty":  "সহজ"}),
        ("ওয়াইফাই পাসওয়ার্ড সুরক্ষিত রাখতে শক্তিশালী পাসওয়ার্ড ব্যবহার করুন।", 
         {"category":  "প্রযুক্তি", "difficulty": "মাঝারি"}),
        ("ডেটা ব্যাকআপ নিতে ক্লাউড স্টোরেজ যেমন গুগল ড্রাইভ ব্যবহার করুন।", 
         {"category": "প্রযুক্তি", "difficulty": "মাঝারি"}),
    ]
    
    # খেলাধুলা (Sports)
    sports_chunks = [
        ("বাংলাদেশ ২০০০ সালে ICC ট্রফি জিতেছিল।", 
         {"category": "খেলাধুলা", "difficulty": "সহজ"}),
        ("ক্রিকেট খেলতে ব্যাট, বল এবং উইকেট প্রয়োজন।", 
         {"category": "খেলাধুলা", "difficulty": "সহজ"}),
        ("ফুটবলে ১১ জন খেলোয়াড় প্রতি দলে থাকে।", 
         {"category": "খেলাধুলা", "difficulty":  "সহজ"}),
        ("শাকিব আল হাসান বাংলাদেশের সেরা অলরাউন্ডার ক্রিকেটার।", 
         {"category":  "খেলাধুলা", "difficulty": "সহজ"}),
        ("২০২২ সালের ফিফা বিশ্বকাপ আর্জেন্টিনা জিতেছে।", 
         {"category":  "খেলাধুলা", "difficulty": "মাঝারি"}),
    ]
    
    # Combine all chunks
    all_chunks = (education_chunks + health_chunks + travel_chunks + 
                  technology_chunks + sports_chunks)
    
    return all_chunks

# Create Vector Store
@st.cache_resource
def create_vector_store():
    """Create FAISS vector store from FAQ data"""
    embedding_model = load_embedding_model()
    all_chunks = prepare_faq_data()
    
    documents = [Document(page_content=text, metadata=meta) 
                 for text, meta in all_chunks]
    
    vector_store = FAISS.from_documents(documents, embedding_model)
    return vector_store, documents

# Metadata Filter Function
def filter_by_metadata(query, category, documents, embedding_model):
    """Filter vector store by category metadata and perform similarity search"""
    st.write(f"**🔍 ক্যাটাগরি ফিল্টার:** {category}")
    
    # Filter documents by category
    filtered_docs = [doc for doc in documents 
                     if doc.metadata['category'] == category]
    
    if not filtered_docs:
        st.warning(f"'{category}' ক্যাটাগরিতে কোনো ডেটা পাওয়া যায়নি।")
        return []
    
    st.write(f"**📚 মোট {len(filtered_docs)} টি ডকুমেন্ট পাওয়া গেছে**")
    
    # Create temporary vector store with filtered documents
    temp_vector_store = FAISS.from_documents(filtered_docs, embedding_model)
    
    # Perform similarity search
    similar_docs = temp_vector_store.similarity_search(query, k=3)
    
    return similar_docs

# --- Setup OpenAI Client ---
def setup_openai_client():
    """Setup OpenAI client with GitHub Models"""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        st.error("NO GITHUB_TOKEN in . env")
        return None
    
    endpoint = "https://models.github.ai/inference"
    model = "openai/gpt-4.1-nano"
    
    client = OpenAI(
        base_url=endpoint,
        api_key=token,
    )
    
    return client, model

# Category Router
def detect_category_llm(question, client, model):
    """Use LLM to automatically detect category from question"""
    system_msg = """তুমি একটি শ্রেণিবিন্যাসকারী এজেন্ট। নিচের প্রশ্নটি পড়ে বলো এটি কোন ক্যাটাগরিতে পড়ে।

    অনুমোদিত ক্যাটাগরি: 
    - শিক্ষা (শিক্ষা, স্কুল, কলেজ, বিশ্ববিদ্যালয়, পড়াশোনা সম্পর্কিত)
    - স্বাস্থ্য (স্বাস্থ্য, রোগ, চিকিৎসা, ওষুধ, পুষ্টি সম্পর্কিত)
    - ভ্রমণ (ভ্রমণ, পর্যটন, স্থান, যাতায়াত সম্পর্কিত)
    - প্রযুক্তি (কম্পিউটার, মোবাইল, ইন্টারনেট, সফটওয়্যার সম্পর্কিত)
    - খেলাধুলা (ক্রিকেট, ফুটবল, খেলা, খেলোয়াড় সম্পর্কিত)
    
    গুরুত্বপূর্ণ নিয়ম:
    - যদি প্রশ্নটি উপরের কোনো ক্যাটাগরিতে স্পষ্টভাবে না মিলে, তাহলে 'অন্যান্য' বলো।
    - এলোমেলো শব্দ, অর্থহীন প্রশ্ন, বা অপ্রাসঙ্গিক প্রশ্নের জন্য 'অন্যান্য' বলো।
    - শুধুমাত্র ক্যাটাগরির নাম বাংলায় এক শব্দে উত্তর দাও।
    - সন্দেহ থাকলে 'অন্যান্য' বলো।"""
    
    try:
        response = client.chat.completions. create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": question}
            ],
            model=model,
            temperature=0,  # More deterministic
            top_p=1.0
        )
        category = response.choices[0].message.content.strip()
        
        # Validate the returned category
        valid_categories = ["শিক্ষা", "স্বাস্থ্য", "ভ্রমণ", "প্রযুক্তি", "খেলাধুলা", "অন্যান্য"]
        
        # Check if response is in valid categories
        if category not in valid_categories:
            st.warning(f"অন্যান্য ক্যাটাগরি পাওয়া গেছে: '{category}'. ডিফল্ট হিসেবে 'অন্যান্য' ব্যবহার করা হচ্ছে।")
            return "অন্যান্য"
        
        return category
        
    except Exception as e:  
        st.error(f"ক্যাটাগরি সনাক্তকরণে সমস্যা: {e}")
        return "অন্যান্য"

# RAG Chain
def ask_faq_bot(user_question, category, documents, embedding_model, client, model):
    """Main RAG function to answer questions"""
    # Filter and retrieve similar documents
    docs = filter_by_metadata(user_question, category, documents, embedding_model)
    
    if not docs:
        return "দুঃখিত, এই বিষয়ে আমার কাছে তথ্য নেই। অনুগ্রহ করে অন্য প্রশ্ন করুন।", []
    
    # Create context from retrieved documents
    context = "\n". join([doc.page_content for doc in docs])
    
    # Display retrieved context
    with st.expander("📄 প্রাসঙ্গিক তথ্য দেখুন"):
        for i, doc in enumerate(docs, 1):
            st.write(f"{i}. {doc.page_content}")
    
    # Generate answer using LLM
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""তুমি একজন সহায়ক বাংলা সহকারী। শুধুমাত্র নিচের প্রাসঙ্গিক তথ্য থেকে উত্তর দাও। 
                    যদি প্রশ্নের উত্তর প্রসঙ্গে না থাকে, বলো 'দুঃখিত, এই বিষয়ে আমার জানা নেই।'
                    
                    প্রসঙ্গ: {context}"""
                },
                {
                    "role": "user",
                    "content": user_question,
                }
            ],
            temperature=0.1,
            top_p=0.9,
            model=model
        )
        answer = response.choices[0].message.content
        return answer, docs
    except Exception as e: 
        return f"উত্তর তৈরিতে সমস্যা হয়েছে: {e}", docs

# Main Streamlit UI
def main():
    # Header
    st.title("🤖 বাংলা FAQ চ্যাটবট")
    st.markdown("### RAG-ভিত্তিক প্রশ্নোত্তর সিস্টেম")
    st.markdown("---")
    
    # Load resources
    with st.spinner("মডেল লোড হচ্ছে..."):
        embedding_model = load_embedding_model()
        vector_store, documents = create_vector_store()
        openai_setup = setup_openai_client()
        
        if openai_setup is None:
            st.stop()
        
        client, model = openai_setup
    
    # Sidebar
    with st.sidebar:
        st. header("⚙️ সেটিংস")
        
        # Category selection mode
        auto_category = st.checkbox("স্বয়ংক্রিয় ক্যাটাগরি সনাক্তকরণ", value=True)
        
        # Manual category selection
        if not auto_category: 
            categories = ["শিক্ষা", "স্বাস্থ্য", "ভ্রমণ", "প্রযুক্তি", "খেলাধুলা"]
            selected_category = st.selectbox("ক্যাটাগরি নির্বাচন করুন:", categories)
        
        st.markdown("---")
        
        # Example questions
        st.subheader("📝 উদাহরণ প্রশ্ন:")
        example_questions = {
            "শিক্ষা": "বিশ্ববিদ্যালয়ে ভর্তির জন্য কী প্রয়োজন?",
            "স্বাস্থ্য": "রক্তচাপ নিয়ন্ত্রণের জন্য কী করব?",
            "ভ্রমণ": "কক্সবাজারে যেতে কত খরচ হবে?",
            "প্রযুক্তি":  "স্মার্টফোনের ব্যাটারি কীভাবে বাঁচাব?",
            "খেলাধুলা": "বাংলাদেশ কবে ICC ট্রফি জিতেছে?"
        }
        
        for question in example_questions.values():
            st.text(f"• {question}")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ চ্যাট মুছুন"):
            st.session_state.chat_history = []
            st. success("চ্যাট মুছে ফেলা হয়েছে!")
    
    # Main chat interface
    st.subheader("💬 প্রশ্ন করুন")
    
    user_question = st.text_input("আপনার প্রশ্ন লিখুন:", 
                                   placeholder="এখানে আপনার প্রশ্ন লিখুন...")
    
    if st.button("উত্তর পান", type="primary"):
        if user_question.strip():
            # Detect category first (outside spinner)
            if auto_category:
                with st.spinner("ক্যাটাগরি সনাক্ত করা হচ্ছে..."):
                    detected_category = detect_category_llm(user_question, client, model)
            
                st.info(f"**সনাক্তকৃত ক্যাটাগরি:** {detected_category}")
            
                # Check if category is valid (outside spinner)
                valid_categories = ["শিক্ষা", "স্বাস্থ্য", "ভ্রমণ", "প্রযুক্তি", "খেলাধুলা"]
                if detected_category not in valid_categories:
                    fallback_message = "এই প্রশ্নটি আমার জ্ঞানের বাইরে। অনুগ্রহ করে অন্য প্রশ্ন করুন।"
                    st.warning(fallback_message)
                
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "category": detected_category,
                        "answer": fallback_message
                    })
                    # Display chat history before stopping
                    if st.session_state.chat_history:
                        st.markdown("---")
                        st.subheader("📜 চ্যাট ইতিহাস")
        
                        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
                            with st.expander(f"প্রশ্ন {i}:  {chat['question'][:50]}... "):
                                st.write(f"**ক্যাটাগরি:** {chat['category']}")
                                st.write(f"**প্রশ্ন:** {chat['question']}")
                                st.write(f"**উত্তর:** {chat['answer']}")
    
                    st.stop()
            
                category = detected_category
            else:   
                category = selected_category
        
            # Get answer
            with st.spinner("উত্তর খুঁজছি..."):
                answer, retrieved_docs = ask_faq_bot(
                    user_question, category, documents, 
                    embedding_model, client, model
                )
        
            # Display answer
            st.markdown("### 🎯 উত্তর:")
            st.success(answer)
        
            # Add to chat history
            st.session_state.chat_history.append({
                "question": user_question,
                "category": category,
                "answer": answer
            })
        else:
            st.warning("অনুগ্রহ করে একটি প্রশ্ন লিখুন।")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📜 চ্যাট ইতিহাস")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"প্রশ্ন {i}:  {chat['question'][: 50]}..."):
                st. write(f"**ক্যাটাগরি:** {chat['category']}")
                st.write(f"**প্রশ্ন:** {chat['question']}")
                st.write(f"**উত্তর:** {chat['answer']}")

if __name__ == "__main__": 
    main()


