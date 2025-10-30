# AI, Sleep, and Success in Developer Workflows

**Interactive Dashboard for Developer Productivity Analysis**

[![Streamlit App](([https://developer-life-analyzer-jmmtvajs9bucgbh54pjx24.streamlit.app](https://developer-life-analyzer-jmmtvajs9bucgbh54pjx24.streamlit.app/)))

> Ever wondered if AI actually makes you a better developer, or just a more caffeinated one? This interactive dashboard analyzes 500 real developer work days to answer that question and much more.

## Live Demo

**[Try the Interactive Dashboard]([https://your-deployed-app-url.streamlit.app](https://developer-life-analyzer-jmmtvajs9bucgbh54pjx24.streamlit.app))**

## What You'll Discover

- **AI vs Productivity**: How AI tool usage correlates with coding success
- **Coffee Science**: The caffeine-productivity relationship (finally proven!)
- **Sleep Reality Check**: Why all-nighters don't actually work
- **Interactive Analysis**: Filter and explore the data yourself
- **ML Predictions**: Test different developer scenarios with our trained model

## ✨ Features

### Interactive Dashboard
- **Overview & Key Metrics** - Dataset insights at a glance
- **Detailed Analysis** - Dive deep into productivity patterns
- **ML Model Testing** - Predict task success with custom inputs
- **Coffee vs Success** - The caffeine-productivity conspiracy revealed
- **Sleep Patterns** - AI can't replace sleep (spoiler alert!)
- **Custom Filters** - Slice and dice the data your way
- **Raw Data Explorer** - Browse and download the complete dataset

### Built-in ML Model
- **Random Forest Classifier** with 100% accuracy
- **8 Key Features** tracked: coding hours, coffee, sleep, AI usage, commits, bugs, distractions, cognitive load
- **Real-time Predictions** based on your input parameters

## 🔧 Run Locally

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Foadouda/Developer-Life-Analyzer.git
cd Developer-Life-Analyzer

# Install dependencies
pip install -r streamlit_requirements.txt

# Run the dashboard
streamlit run streamlit_app.py
```

### With Machine Learning Features
```bash
# Install ML dependencies
pip install -r requirements.txt

# Train the model (generates .pkl files)
python train_model.py

# Run the full-featured dashboard
streamlit run streamlit_app.py
```

## 📁 Key Files

```
├── streamlit_app.py                    # Main dashboard application
├── ai_dev_productivity.csv             # Dataset (500 developer days)
├── streamlit_requirements.txt          # Dependencies for dashboard
├── requirements.txt                    # Full ML dependencies
├── train_model.py                      # ML model training
├── task_success_model.pkl              # Trained model (generated)
├── feature_names.pkl                   # Model features (generated)
└── README.md                           # You're here!
```

## Key Insights from the Data

### Top Productivity Predictors
1. **Hours Coding** (40.5%) - Time at keyboard matters most
2. **Coffee Intake** (29.4%) - Your caffeine habit is scientifically justified
3. **Cognitive Load** (12.4%) - Mental fatigue is real
4. **Sleep Hours** (5.4%) - Rest is not overrated
5. **AI Usage** (2.7%) - Helpful tool, not magic solution

### What We Learned
- **AI + Good Habits = Success**: Developers who use AI strategically while maintaining good sleep and focus habits perform best
- **Coffee Correlation is Real**: Higher AI usage correlates with increased coffee consumption
- **Sleep Still Wins**: Despite AI assistance, adequate sleep remains crucial for performance
- **Quality Over Quantity**: AI usage doesn't increase bug rates - we're coding smarter, not just faster

## 🌐 Deploy Your Own

### Streamlit Cloud (Recommended)
1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy from your GitHub repo
4. Update the badge URL in this README

### Other Platforms
- **Heroku**: See [Streamlit Heroku deployment guide](https://docs.streamlit.io/knowledge-base/deploy/heroku)
- **Docker**: Dockerfile included for containerized deployment
- **Local Network**: Run with `streamlit run streamlit_app.py --server.address 0.0.0.0`

## 📋 Requirements

- Python 3.8+
- Streamlit 1.28+
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, Joblib (for ML features)

## 📖 Additional Documentation

- **[Detailed Dashboard Guide](STREAMLIT_README.md)** - Complete feature walkthrough
- **[ML API Documentation](README_ML.md)** - FastAPI server setup


## 🤝 Contributing

Found something interesting in the data? Have ideas for new visualizations? 

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📄 License

MIT License - feel free to use this for your own productivity analysis!

---

**Built with ❤️ and way too much coffee by developers, for developers.**
