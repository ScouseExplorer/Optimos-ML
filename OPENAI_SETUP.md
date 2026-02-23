# 🚀 OpenAI Integration Setup Guide

## Quick Start (5 minutes)

### Step 1: Get Your OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

### Step 2: Add API Key to Your Project

1. Copy the example file:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Open `.env` file and replace `your_openai_api_key_here` with your actual key:
   ```
   OPENAI_API_KEY=sk-proj-abc123...
   ```

3. Save the file

### Step 3: Test It!

```powershell
# Start the API server
C:/Users/akinl/LinkedIn-AI/.venv/Scripts/python.exe -m uvicorn api.app:app --reload

# In another terminal, test the refinement
C:/Users/akinl/LinkedIn-AI/.venv/Scripts/python.exe test_refine.py
```

## 💰 Pricing

**GPT-3.5-Turbo** (Recommended for testing):
- $0.0015 per 1K tokens input
- $0.002 per 1K tokens output
- **~$0.001-0.003 per post** (basically free)

**GPT-4** (Higher quality):
- $0.03 per 1K tokens
- **~$0.02-0.05 per post**

## 🎯 API Endpoints

### 1. POST /refine/post
Refine a post with AI based on ML insights

**Request:**
```json
{
  "content": "Your post here...",
  "style": "professional",  // or "casual", "inspirational"
  "variants": 1  // Generate 1-3 versions
}
```

**Response:**
```json
{
  "original": "...",
  "refined": ["improved version 1", "improved version 2"],
  "analysis": {
    "linear_score": 45.2,
    "viral_probability": 68.5
  },
  "improvements": ["Add question", "Add hashtags"],
  "estimated_improvement": {
    "version_1": {
      "engagement_score": 72.8,
      "improvement": +27.6
    }
  }
}
```

### 2. POST /refine/compare
Side-by-side comparison of original vs refined

**Request:**
```json
{
  "content": "Your post..."
}
```

**Response:**
```json
{
  "comparison": {
    "original": { "engagement_score": 45, ... },
    "refined": { "engagement_score": 73, "improvement": +28 }
  },
  "recommendation": "Use refined version"
}
```

## 🔧 Configuration

Edit `.env` to customize:

```env
# Model selection
OPENAI_MODEL=gpt-3.5-turbo  # Fast & cheap
# OPENAI_MODEL=gpt-4          # Best quality

# Creativity (0-1)
OPENAI_TEMPERATURE=0.7  # 0.3=conservative, 0.9=creative

# Max response length
OPENAI_MAX_TOKENS=500  # 300-800 recommended
```

## ⚠️ Important Notes

1. **Never commit `.env` to Git** (already in .gitignore)
2. **Keep your API key secret**
3. **Monitor usage** at https://platform.openai.com/usage
4. **Set spending limits** in OpenAI dashboard

## 🎨 Style Options

- **professional**: Executive tone, thought leadership
- **casual**: Conversational, relatable, friendly
- **inspirational**: Motivational, action-oriented

## 🧪 Testing Without API Key

If you don't have an API key yet, the app will still work but refinement endpoints will return an error. All ML prediction endpoints work independently.

## 📖 Full Documentation

Visit http://localhost:8000/docs when the server is running.
