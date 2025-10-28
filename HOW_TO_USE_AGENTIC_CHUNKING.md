

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Deep Config Mode (Step-by-Step)](#deep-config-mode)
4. [Config-1 Mode (Step-by-Step)](#config-1-mode)
5. [Understanding Strategies](#understanding-strategies)
6. [Use Cases & Examples](#use-cases--examples)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Prerequisites

### 1. **Set Up Gemini API Key** (Required!)

Agentic chunking requires Google Gemini API. Get your key from: https://makersuite.google.com/app/apikey

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Verify it's set:**
```powershell
# Windows
echo $env:GEMINI_API_KEY

# Linux/Mac
echo $GEMINI_API_KEY
```

### 2. **Start the Application**

**Terminal 1 - Backend:**
```bash
cd iChunk_Op
python main.py
```
Wait for: `INFO:     Uvicorn running on http://127.0.0.1:8001`

**Terminal 2 - Frontend:**
```bash
cd ichunk-react
npm run dev
```
Open browser to: `http://localhost:5173`

---

## 🚀 Quick Start

### **Fastest Way to Try Agentic Chunking:**

1. Open browser → `http://localhost:5173`
2. Select **"Config-1 Mode"**
3. Upload a CSV file
4. Go to **"Chunking"** tab
5. Select **"🤖 Agentic (AI-Powered)"** from dropdown
6. Click **"Run Processing"**
7. Done! AI will analyze and chunk your data intelligently

---

## 📊 Deep Config Mode (Step-by-Step)

Deep Config gives you the most control over agentic chunking.

### **Step 1: Select Mode**
```
┌─────────────────────────────────┐
│  Choose Processing Mode         │
│  ○ Fast Mode                    │
│  ○ Config-1 Mode                │
│  ● Deep Config  ← Select this   │
│  ○ Campaign Mode                │
└─────────────────────────────────┘
```

### **Step 2: Upload Your Data**
```
┌─────────────────────────────────┐
│  📁 Upload CSV File             │
│  [Choose File] or Drag & Drop   │
│                                 │
│  OR                             │
│                                 │
│  🗄️ Connect to Database         │
│  [Database Connection]          │
└─────────────────────────────────┘
```

**Click "Upload & Analyze"**

### **Step 3: Data Type Conversion**
- Review detected data types
- Adjust if needed
- Click **"Apply Type Conversion"**

### **Step 4: Null Handling**
- Review null values
- Choose strategies (remove/mean/mode/custom)
- Click **"Apply Null Handling"**

### **Step 5: Duplicate Removal**
- Review duplicates
- Select columns to check
- Click **"Remove Duplicates"**

### **Step 6: Text Preprocessing**
- Choose preprocessing options:
  - ✓ Remove HTML tags
  - ✓ Lowercase
  - ✓ Remove stopwords
  - ✓ Lemmatization
- Click **"Apply Preprocessing"**

### **Step 7: Chunking** ⭐ **THIS IS WHERE AGENTIC HAPPENS!**

#### **7.1 Select Agentic Chunking:**
```
┌─────────────────────────────────────────────┐
│  Chunking Method:                           │
│  [🤖 Agentic (AI-Powered) ▼]               │
└─────────────────────────────────────────────┘
```

**You'll see 5 options:**
- Fixed Size
- Recursive
- Semantic Clustering
- Document Based
- **🤖 Agentic (AI-Powered)** ← **SELECT THIS**

#### **7.2 Configure Agentic Strategy:**

Once selected, you'll see this panel:

```
┌─────────────────────────────────────────────┐
│  🤖 AI-Powered Agentic Chunking             │
├─────────────────────────────────────────────┤
│                                             │
│  Chunking Strategy:                         │
│  [🤖 Auto (AI Decides) ▼]                  │
│                                             │
│  Options:                                   │
│  • 🤖 Auto (AI Decides Best Strategy)      │
│  • Schema-Aware (Analyzes table structure) │
│  • Entity-Centric (Groups by entities)     │
│                                             │
│  User Context (Optional):                   │
│  [Type your context here...............]    │
│                                             │
│  💡 How Agentic Chunking Works:             │
│  • AI analyzes your data structure          │
│  • Identifies entities (users, products)    │
│  • Decides optimal grouping strategy        │
│  • Preserves context and semantic meaning   │
│                                             │
│  ⚠️ Requires GEMINI_API_KEY environment var │
└─────────────────────────────────────────────┘
```

#### **7.3 Choose Your Strategy:**

**Option A: Auto (Recommended for beginners)**
```
Strategy: [🤖 Auto (AI Decides) ▼]
Context: [Leave empty or add: "Analyze customer data"]
```
✅ AI will automatically choose the best strategy

**Option B: Schema-Aware**
```
Strategy: [Schema-Aware ▼]
Context: ["Focus on product categories and prices"]
```
✅ Best for structured data with clear column relationships

**Option C: Entity-Centric**
```
Strategy: [Entity-Centric ▼]
Context: ["Group by customer ID and transaction history"]
```
✅ Best for data with clear entities (users, products, companies)

#### **7.4 Click "Process Chunking"**

**What happens next:**
```
🔄 Processing...
├── AI analyzing your data schema
├── Identifying column types and relationships
├── Detecting entities (if any)
├── Selecting optimal chunking strategy
├── Creating intelligent chunks
└── ✅ Done! X chunks created
```

### **Step 8: Embedding**
- Choose embedding model (default: paraphrase-MiniLM-L6-v2)
- Click **"Generate Embeddings"**

### **Step 9: Storage**
- Choose storage (FAISS or ChromaDB)
- Click **"Store Vectors"**

### **Step 10: Retrieval & Export**
- Now you can search your intelligently chunked data!
- Export results as needed

---

## ⚡ Config-1 Mode (Step-by-Step)

Config-1 is faster with fewer steps.

### **Step 1: Select Config-1 Mode**
```
┌─────────────────────────────────┐
│  ○ Fast Mode                    │
│  ● Config-1 Mode  ← Select      │
│  ○ Deep Config                  │
│  ○ Campaign Mode                │
└─────────────────────────────────┘
```

### **Step 2: Upload Data**
- Upload CSV or connect to database
- Click **"Upload"**

### **Step 3: Go to Chunking Tab**
```
┌─────────────────────────────────────────┐
│  [Preprocessing] [Chunking*] [Embedding] │
└─────────────────────────────────────────┘
```
Click on **"Chunking"** tab

### **Step 4: Select Agentic Chunking**
```
┌─────────────────────────────────────────┐
│  Chunking Method:                       │
│  [🤖 Agentic (AI-Powered) ▼]           │
└─────────────────────────────────────────┘
```

### **Step 5: Configure Strategy**
```
┌─────────────────────────────────────────┐
│  Chunking Strategy:                     │
│  [🤖 Auto (AI Decides) ▼]              │
│                                         │
│  User Context (Optional):               │
│  [e.g., 'Analyze sales by region']     │
└─────────────────────────────────────────┘
```

### **Step 6: Run Processing**
Click the big **"Run Processing"** button at the bottom

**That's it!** Config-1 handles preprocessing, chunking, embedding, and storage all at once.

---

## 🎓 Understanding Strategies

### **🤖 Auto (AI Decides)**

**What it does:**
- Analyzes your entire dataset
- Examines column types, cardinality, relationships
- Automatically selects the best strategy
- No manual configuration needed

**Best for:**
- First-time users
- Unknown data structures
- Quick experiments

**Example:**
```
Strategy: Auto
Context: "Analyze e-commerce transactions"

AI Decision: "Detected user_id and product_id columns with 
high cardinality. Selecting Entity-Centric strategy to 
group by user_id."
```

---

### **📋 Schema-Aware**

**What it does:**
- Analyzes table structure (columns, data types)
- Identifies relationships between columns
- Groups related columns together
- Preserves hierarchical structure

**Best for:**
- Structured data with clear schemas
- Data with parent-child relationships
- Multi-level categorization

**Example CSV:**
```csv
category,subcategory,product,price,stock
Electronics,Phones,iPhone,999,50
Electronics,Phones,Samsung,799,30
Clothing,Shirts,T-Shirt,29,100
```

**AI will:**
1. Identify hierarchy: category → subcategory → product
2. Group by category first
3. Create chunks preserving the hierarchy

**User Context Examples:**
- "Focus on product categories and inventory"
- "Analyze pricing structure by category"
- "Group by department and subdepartment"

---

### **👤 Entity-Centric**

**What it does:**
- Identifies primary entities (users, products, companies)
- Groups all rows related to the same entity
- Preserves entity context across chunks

**Best for:**
- Customer data (group by customer_id)
- Product catalogs (group by product_id)
- Company records (group by company_name)
- Transaction logs (group by user_id)

**Example CSV:**
```csv
user_id,transaction_date,product,amount
U001,2024-01-01,Laptop,1200
U001,2024-01-05,Mouse,25
U002,2024-01-02,Keyboard,75
U002,2024-01-03,Monitor,300
```

**AI will:**
1. Identify `user_id` as primary entity
2. Create chunks: All U001 transactions together, all U002 together
3. Preserve user purchase history

**User Context Examples:**
- "Group by customer for purchase analysis"
- "Analyze user behavior patterns"
- "Group by product for sales trends"

---

## 💡 Use Cases & Examples

### **Use Case 1: E-Commerce Customer Data**

**Your Data:**
```csv
customer_id,name,email,purchase_date,product,price,category
C001,John,john@email.com,2024-01-01,Laptop,1200,Electronics
C001,John,john@email.com,2024-01-05,Mouse,25,Accessories
C002,Jane,jane@email.com,2024-01-02,Shirt,50,Clothing
```

**Recommended Setup:**
```
Mode: Config-1 or Deep Config
Strategy: Entity-Centric
Context: "Group by customer for purchase history analysis"
```

**Result:**
- Chunk 1: All C001 purchases together
- Chunk 2: All C002 purchases together
- Perfect for customer behavior analysis!

---

### **Use Case 2: Product Catalog**

**Your Data:**
```csv
category,brand,product,price,specs,rating
Electronics,Apple,iPhone 15,999,128GB 5G,4.8
Electronics,Samsung,Galaxy S24,899,256GB 5G,4.7
Clothing,Nike,Running Shoes,120,Size 10,4.5
```

**Recommended Setup:**
```
Mode: Deep Config
Strategy: Schema-Aware
Context: "Organize by category and brand hierarchy"
```

**Result:**
- Chunks preserve category → brand → product structure
- Easy to retrieve "all electronics" or "all Apple products"

---

### **Use Case 3: Sales Data**

**Your Data:**
```csv
region,store_id,date,product,quantity,revenue
North,S001,2024-01-01,Widget,100,5000
North,S001,2024-01-02,Gadget,50,2500
South,S002,2024-01-01,Widget,80,4000
```

**Recommended Setup:**
```
Mode: Config-1
Strategy: Auto
Context: "Analyze sales by region and store"
```

**Result:**
- AI automatically detects region/store hierarchy
- Creates chunks optimized for regional analysis

---

### **Use Case 4: Unknown Data Structure**

**Your Data:**
- You just received a CSV
- Don't know the structure
- Need quick insights

**Recommended Setup:**
```
Mode: Config-1 (fastest)
Strategy: Auto (let AI decide)
Context: "General analysis" or leave empty
```

**Result:**
- AI analyzes everything
- Chooses best strategy automatically
- You get intelligent chunks without manual work!

---

## 🎯 Pro Tips

### **Tip 1: Use Descriptive Context**

**❌ Bad:**
```
Context: [empty]
```

**✅ Good:**
```
Context: "Analyze customer purchase patterns by region"
Context: "Group products by category for inventory management"
Context: "Focus on user behavior and engagement metrics"
```

**Why?** The AI uses your context to make better decisions!

---

### **Tip 2: Start with Auto, Then Refine**

**First Run:**
```
Strategy: Auto
Context: "General analysis"
```

**Check the results, then refine:**
```
Strategy: Entity-Centric (if you see clear entities)
Context: "Group by user_id for behavior analysis"
```

---

### **Tip 3: Match Strategy to Your Goal**

| Your Goal | Best Strategy | Example Context |
|-----------|---------------|-----------------|
| Customer analysis | Entity-Centric | "Group by customer_id" |
| Product catalog | Schema-Aware | "Organize by category hierarchy" |
| Unknown data | Auto | "General analysis" |
| Time-series data | Auto | "Analyze trends over time" |
| Multi-level data | Schema-Aware | "Preserve department structure" |

---

## 🔧 Troubleshooting

### **Problem 1: "Agentic option not showing"**

**Solution:**
1. Refresh browser (Ctrl + F5)
2. Check if React dev server is running
3. Look in the correct place:
   - **Deep Config:** Step 7 (Chunking)
   - **Config-1:** Chunking tab

---

### **Problem 2: "GEMINI_API_KEY not set" error**

**Solution:**
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"

# Verify
echo $env:GEMINI_API_KEY

# Restart backend
cd iChunk_Op
python main.py
```

---

### **Problem 3: "Agentic chunking failed"**

**Possible causes:**
1. **Invalid API key** → Check your Gemini API key
2. **No internet** → Agentic needs API access
3. **Empty dataset** → Upload data first
4. **API quota exceeded** → Check Gemini quota

**Solution:**
```bash
# Check backend logs
# Look for errors like:
# "Gemini API error: ..."
# "Failed to analyze schema: ..."
```

---

### **Problem 4: "Chunks don't look right"**

**Solution:**
1. Try a different strategy
2. Add more specific context
3. Check your data quality:
   - Are column names descriptive?
   - Is data properly formatted?
   - Any missing values?

---

## 📊 Visual Guide

### **Where to Find Agentic in Deep Config:**

```
Deep Config Mode
├── Step 1: Upload ✓
├── Step 2: Type Conversion ✓
├── Step 3: Null Handling ✓
├── Step 4: Duplicates ✓
├── Step 5: Text Preprocessing ✓
├── Step 6: Chunking ⭐ YOU ARE HERE
│   ├── Chunking Method Dropdown
│   │   ├── Fixed Size
│   │   ├── Recursive
│   │   ├── Semantic Clustering
│   │   ├── Document Based
│   │   └── 🤖 Agentic (AI-Powered) ← SELECT
│   │
│   └── Agentic Configuration Panel (appears when selected)
│       ├── Strategy Selector
│       ├── User Context Input
│       └── Info Panel
│
├── Step 7: Embedding
└── Step 8: Storage
```

### **Where to Find Agentic in Config-1:**

```
Config-1 Mode
├── Upload Data ✓
├── Configuration Tabs
│   ├── [Preprocessing]
│   ├── [Chunking] ⭐ CLICK HERE
│   │   ├── Chunking Method Dropdown
│   │   │   └── 🤖 Agentic (AI-Powered) ← SELECT
│   │   └── Agentic Configuration Panel
│   ├── [Embedding]
│   └── [Storage]
└── Run Processing Button
```

---

## 🎉 Quick Reference Card

**Fastest Way to Use Agentic Chunking:**

1. ✅ Set `GEMINI_API_KEY` environment variable
2. ✅ Start backend: `python main.py`
3. ✅ Start frontend: `npm run dev`
4. ✅ Open browser: `http://localhost:5173`
5. ✅ Select **Config-1 Mode**
6. ✅ Upload CSV
7. ✅ Go to **Chunking** tab
8. ✅ Select **🤖 Agentic (AI-Powered)**
9. ✅ Choose **Auto** strategy
10. ✅ Click **Run Processing**

**Done! Your data is now intelligently chunked by AI!** 🚀

---

## 📚 Additional Resources

- **Backend Code:** `iChunk_Op/backend_agentic.py`
- **API Endpoint:** `/deep_config/chunk` (POST)
- **LLM Config:** `iChunk_Op/llm_config.yaml` (agentic_chunking profile)
- **Documentation:** `AGENTIC_CHUNKING_IMPLEMENTATION_GUIDE.md`

---

## 🆘 Need Help?

**Common Questions:**

**Q: Which mode should I use?**
A: Config-1 for speed, Deep Config for control

**Q: Which strategy should I choose?**
A: Start with Auto, it's smart!

**Q: Do I need context?**
A: Optional but recommended for better results

**Q: How much does Gemini API cost?**
A: Check Google AI Studio pricing (has free tier!)

**Q: Can I use without API key?**
A: No, agentic chunking requires Gemini API

---

## ✅ Success Checklist

Before using agentic chunking, ensure:

- [ ] GEMINI_API_KEY is set in environment
- [ ] Backend is running on port 8001
- [ ] Frontend is running on port 5173
- [ ] You can see "🤖 Agentic (AI-Powered)" in dropdown
- [ ] You have a CSV file ready to upload
- [ ] Internet connection is active (for API calls)

**If all checked, you're ready to go!** 


