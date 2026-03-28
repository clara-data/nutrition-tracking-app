# Nutrition Tracking Application

This project is a nutrition tracking application that allows users to enter foods, match them with USDA food data, and generate daily nutrition summaries through a dashboard.

## Main Features
- Natural language meal input
- Food matching using USDA data
- Daily nutrition summary
- Dashboard display
- LLM-based dietary recommendations

## Project Structure
- `Rbi App.py`: main Streamlit application
- `rbi_pipeline.py`: pipeline script
- `backend/`: core logic for matching, processing, and recommendations
- `requirements.txt`: required Python packages
- `user_data/`: stores user input and output files

## Setup Instructions

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama and download the model
```bash
ollama pull llama3
```

# Run the Streamlit app
- streamlit run "Personal_Nutrition_Tracking_App.py"

# Notes
- The application runs locally.
- USDA data should be placed in the USDA_data/ folder.
- The LLM-related functions require Ollama to be installed.

# Demo Video
A short demo of the application is available here:
https://drive.google.com/file/d/1MrvP3K0Chuo4MWJ2Y0FZb1FcKPWQoDsI/view?usp=sharing