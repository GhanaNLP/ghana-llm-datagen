# 🇬🇭 Ghana LLM Dataset — Volunteer Campaign

We're generating a high-quality Ghanaian conversational AI dataset from news articles and research papers. Volunteers run API calls on their machines for ~24 hours using a code we provide.

**No coding experience needed. One command does everything.**

---

## ⚡ Quick Start (Volunteers)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ghana-llm-datagen.git
cd ghana-llm-datagen
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run with your code
```bash
python run.py --code YOUR_VOLUNTEER_CODE
```

> Your volunteer code is sent to you by the project owner. It looks like a long string of letters and numbers.

**That's it.** The script will:
- 🔑 Decode your API key and batch assignment from the code
- ⬇️ Download your portion of the dataset automatically
- ⚙️ Generate conversations with a live progress bar
- 💾 Save results locally with **auto-resume** if interrupted
- 📤 Tell you how to submit when done

---

## 🔁 If Your Run Gets Interrupted

Just re-run the same command:
```bash
python run.py --code YOUR_VOLUNTEER_CODE
```
It resumes exactly where it left off. Nothing is lost.

---

## 📤 Submitting Your Results

When the run finishes, it will print submission instructions. You'll [open a GitHub issue](../../issues/new?template=result_submission.md) and attach your `.jsonl` results file.

---

## ❓ FAQ

**Q: How long will it take?**  
Typically 6–16 hours depending on batch size. You can leave it running overnight.

**Q: Is the code safe? Is it an API key?**  
Your code is a volunteer-specific token that encodes your batch assignment and a temporary API key. Don't share it with others.

**Q: Can I run on a server / VM / cloud?**  
Yes! Any machine with Python 3.10+ and internet access works.

**Q: What if I see lots of warnings or errors?**  
Some failures are normal — the script retries automatically. As long as the progress bar is moving, you're fine.

**Q: What GPU / hardware do I need?**  
None. All computation happens on NVIDIA's API servers. You just need a normal laptop or PC.

---

## 📁 Repo Structure

```
ghana-llm-datagen/
├── run.py                     ← Volunteers run this
├── requirements.txt
├── scripts/
│   ├── admin_setup.py         ← Owner: generate volunteer codes
│   ├── split_csv.py           ← Owner: split large CSVs into batches
│   └── merge_results.py       ← Owner: merge all submitted results
└── .github/
    └── ISSUE_TEMPLATE/
        └── result_submission.md
```

---

## 🙏 Contributing

All volunteers will be credited in the final dataset release. Thank you for helping build AI resources for Ghana! 🇬🇭
