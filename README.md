# Ghana Chat Data Generation

**PROJECT STATUS: COMPLETE!**

Thanks to the incredible efforts of the **Ghana NLP Community** and our dedicated volunteers, the data generation phase of this project is officially finished!

Together, we successfully processed millions of rows of Ghanaian news, international news reported from a Ghanaian perspective, and research data through Llama 3.1 70B to create a high-quality, multi-turn conversational dataset.

**Access the Final Dataset:**

[**Ghana-Chat on Hugging Face**](https://huggingface.co/datasets/ghananlpcommunity/ghana-chat)

A massive thank you to everyone who donated their time and compute to help build this foundational AI resource for Ghana!

## Archive: Original Volunteer Instructions

*The instructions below are preserved for historical reference, transparency, and reproducibility.*

We're generating a high-quality Ghanaian conversational AI dataset from news articles and research papers.

**No coding experience needed. One command does everything.**

### Quick Start

**1. Clone the repo**

```
git clone [https://github.com/YOUR_USERNAME/ghana-llm-datagen.git](https://github.com/YOUR_USERNAME/ghana-llm-datagen.git)
cd ghana-llm-datagen
```

**2. Run the code to generate LLM data**

```
python run.py --code YOUR_VOLUNTEER_CODE
```

Or for faster processing, run the async version:

```
python run_async.py --code YOUR_VOLUNTEER_CODE
```

> Your volunteer code is sent to you by the project owner.

**That's it.** The script will:

- Download your portion of the dataset automatically
- Generate conversations with a live progress bar
- Save results locally with **auto-resume** if interrupted
- Tell you how to submit when done

### If Your Run Gets Interrupted

Just re-run the same command:

```
python run.py --code YOUR_VOLUNTEER_CODE
```

Using async version:

```
python run_async.py --code YOUR_VOLUNTEER_CODE
```

It resumes exactly where it left off. Nothing is lost.

### Submitting Your Results

When the run finishes, it will print submission instructions. You'll open a GitHub issue and attach your `.jsonl` results file.

We have a dashboard to track progress [here](https://huggingface.co/spaces/ghananlpcommunity/ghana-llm-datagen-tracker).

### FAQ

**Q: How long will it take?** Typically 80–100 hours depending on the api server speed. You can stop and resume the run with progress preserved.

**Q: How long will it take for Async version?** Typically 48–72 hours depending on the api server speed. You can stop and resume the run with progress preserved.

**Q: Is the code safe? Is it an API key?** Your code is a volunteer-specific token that encodes your batch assignment and a temporary API key.

**Q: Can I run on a server / VM / cloud?** Yes! Any machine with Python 3.10+ and internet access works.

**Q: What if I see lots of warnings or errors?** Some failures are normal — the script retries automatically. As long as the progress bar is moving, you're fine.

**Q: What GPU / hardware do I need?** None. All computation happens on NVIDIA's API servers. You just need a normal laptop or PC.

### Contributing

All volunteers are credited in the final dataset release. Thank you for helping build AI resources for Ghana!

