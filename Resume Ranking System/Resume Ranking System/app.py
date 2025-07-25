from flask import Flask, request, render_template, redirect, url_for, flash
import os
import pdfplumber
import torch
import re
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.secret_key = 'your_secret_key'  

# Load model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('F:\FYP Project\Resume Ranking System\Resume Ranking System\saved_model')
model = DistilBertModel.from_pretrained('F:\FYP Project\Resume Ranking System\Resume Ranking System\saved_model')

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text.strip()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", 
                      max_length=512, 
                      truncation=True,
                      padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy().flatten()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle login logic here
        flash('Login functionality not implemented yet!', 'info')
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Handle signup logic here
        flash('Signup functionality not implemented yet!', 'info')
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/rank', methods=['GET', 'POST'])
def rank_resumes():
    if request.method == 'GET':
        # Render the upload form
        return render_template('rank_resumes.html')
    
    elif request.method == 'POST':
        # Handle form submission
        job_desc = request.form['job_desc']
        resumes = request.files.getlist('resumes')
        
        # Process job description
        cleaned_jd = clean_text(job_desc)
        jd_embedding = get_embedding(cleaned_jd)
        
        # Process resumes
        results = []
        for resume in resumes:
            if resume.filename.endswith('.pdf'):
                # Save temp file
                filename = os.path.join(app.config['UPLOAD_FOLDER'], resume.filename)
                resume.save(filename)
                
                # Extract and clean text
                text = extract_text_from_pdf(filename)
                cleaned_text = clean_text(text)
                
                # Get embedding and similarity
                resume_embedding = get_embedding(cleaned_text)
                similarity = cosine_similarity([jd_embedding], [resume_embedding])[0][0]
                
                results.append({
                    'filename': resume.filename,
                    'score': round(similarity, 4)
                })
                
                # Clean up
                os.remove(filename)
            else:
                flash(f"Unsupported file format for {resume.filename}. Only PDFs are accepted.", 'error')
        
        if not results:
            flash("No valid resumes were uploaded.", 'error')
            return redirect(request.url)
        
        # Sort results
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return render_template('results.html', 
                             job_desc=job_desc,
                             results=sorted_results)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)




