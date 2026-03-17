# LinkedIn Growth AI 🚀

An AI-powered LinkedIn content optimization tool that uses machine learning to predict post engagement and generate high-performing content suggestions.

## 🌟 Features

- **Engagement Prediction**: ML models trained on real LinkedIn data to predict likes, comments, and shares
- **AI Content Generation**: OpenAI-powered content suggestions optimized for maximum engagement
- **Feature Analysis**: Detailed analysis of what makes LinkedIn posts successful
- **REST API**: Easy-to-use API for integrating predictions into your workflow
- **Real-time Insights**: Get instant feedback on your content before posting

## 🏗️ Architecture

```
LinkedIn-growth/
├── api/              # Flask REST API
├── data/             # Training data and datasets
├── ml/               # Machine learning models
├── nlp/              # Text processing and feature extraction
└── models/           # Trained model artifacts
```

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Git

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ScouseExplorer/Optimos-ML.git
cd Optimos-ML
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example environment file
copy .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

### 5. Train Models (Optional)

```bash
python train_models.py
```

### 6. Start the API

```bash
# Using PowerShell script (Windows)
.\start_api.ps1

# Or directly with Python
python api/app.py
```

The API will be available at `http://localhost:5000`

## 🔧 API Usage

### Predict Engagement

```bash
POST /predict
Content-Type: application/json

{
  "content": "Your LinkedIn post content here",
  "industry": "Technology",
  "post_type": "Article"
}
```

**Response:**
```json
{
  "predictions": {
    "likes": 150,
    "comments": 25,
    "shares": 10
  },
  "confidence": 0.85,
  "suggestions": [
    "Add more specific examples",
    "Include a call-to-action"
  ]
}
```

### Get AI Suggestions

```bash
POST /refine
Content-Type: application/json

{
  "content": "Your draft post",
  "target_metric": "engagement",
  "tone": "professional"
}
```

## 📊 Model Performance

The ML models are trained on LinkedIn post data with the following metrics:

- **Engagement Prediction**: R² Score > 0.75
- **Feature Importance**: Analyzes 20+ content features
- **Text Processing**: Advanced NLP with sentiment analysis

## 🧪 Testing

```bash
# Test the API
python test_api.py

# Test content refinement
python test_refine.py
```

## 📈 Training Your Own Models

1. **Collect Data**: Follow the guide in `DATA_COLLECTION_GUIDE.md`
2. **Prepare Dataset**: Add your LinkedIn posts to `data/linkedin_posts.csv`
3. **Train Models**: Run `python train_models.py`
4. **Update Features**: Use `python update_training_data.py` to refresh feature extraction

## 📁 Key Files

- `api/app.py` - Flask REST API server
- `ml/regression_models.py` - ML model training and prediction
- `nlp/feature_extractor.py` - NLP feature extraction
- `nlp/text_processor.py` - Text preprocessing
- `train_models.py` - Model training pipeline
- `requirements.txt` - Python dependencies

## 🤖 AI Integration

This project uses OpenAI's GPT models for:
- Content refinement suggestions
- Tone optimization
- Engagement enhancement recommendations

See `OPENAI_INTEGRATION_COMPLETE.md` for detailed setup instructions.

## 🔐 Security

- Never commit your `.env` file
- Keep your OpenAI API key secure
- Rotate API keys regularly
- Use environment variables for all secrets

## 📝 Data Collection

Want to train on your own data? Check out `DATA_COLLECTION_GUIDE.md` for:
- Manual data collection templates
- Best practices for gathering LinkedIn metrics
- Data formatting guidelines

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **ML**: scikit-learn, pandas, numpy
- **NLP**: NLTK, spaCy, TextBlob
- **AI**: OpenAI GPT-5
- **Data**: CSV-based training data

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT API
- scikit-learn community
- LinkedIn for inspiration

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check existing documentation
- Review the setup guides

## 🎯 Roadmap

- [ ] Add support for image analysis
- [ ] Implement A/B testing features
- [ ] Create web dashboard
- [ ] Add batch processing
- [ ] Support multiple languages
- [ ] Integration with LinkedIn API

---

Made with ❤️ for LinkedIn content creators
