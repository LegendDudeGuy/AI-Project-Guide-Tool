from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_caching import Cache
from flask_socketio import SocketIO
import ollama
import hashlib
import threading

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'

# Setup caching
cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 0})  # Set default timeout to 10 minutes

# Setup SocketIO for asynchronous processing
socketio = SocketIO(app, async_mode='eventlet')

# Mock database as a dictionary
users_db = {
    'testuser': {
        'username': 'testuser',
        'password': hashlib.sha256('testpassword'.encode()).hexdigest(),
        'problem_statements': {
            "Maker": [
                {"title": "Checkpoint 1", "steps": ["a. use me", "b. use you"]},
                {"title": "Checkpoint 2", "steps": ["a. use this"]},
                {"title": "Checkpoint 3", "steps": ["a. use me", "b. use you"]},
                {"title": "Checkpoint 4", "steps": ["a. use this"]},
                {"title": "Checkpoint 5", "steps": ["a. use me", "b. use you"]},
                {"title": "Checkpoint 6", "steps": ["a. use this"]}
            ],
            "Builder": [
                {"title": "Checkpoint 1", "steps": ["a. don't use this"]},
                {"title": "Checkpoint 2", "steps": ["a. use this"]},
                {"title": "Checkpoint 3", "steps": ["a. use me", "b. use you"]},
                {"title": "Checkpoint 4", "steps": ["a. use this"]}
            ]
        },
        'skill': "Intermediate"
    }
}

# Temporary database for form selection
form_selections_db = {}

@cache.memoize(timeout=0)  # Cache for 10 minutes
def generate_project_ideas(domain):
    prompt = (
        f"Generate exactly two project ideas in the domain of {domain}. "
        "Each project idea should include exactly 5 key points describing the project. "
        "Provide the ideas in plain text without any formatting such as headings, asterisks, or bullet points, don't list the project number or project just the project name is enough. "
        "Just list the project ideas and their key points in a simple, unformatted manner, do not use any opening or closing statements."
    )
    response = ollama.generate(model='llama3.1', prompt=prompt)
    project_ideas = response.get('response', '')  # Handle missing response gracefully
    return project_ideas.strip().split('\n\n')

@cache.memoize(timeout=0)  # Cache for 10 minutes
def generate_guide(selected_idea, domain):
    prompt = (
        f"Create a step-by-step guide for the following project idea in the domain of {domain}: "
        f"{selected_idea}. Include clear and concise instructions for each step, covering the key aspects of the project, limit it to 5 steps, keep it short. "
        "The guide should be formatted in plain text without any special formatting or bullet points."
    )
    response = ollama.generate(model='llama3.1', prompt=prompt)
    guide = response.get('response', '')  # Handle missing response gracefully
    return guide.strip()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login_signup', methods=['GET', 'POST'])
def login_signup():
    if request.method == 'POST':
        action = request.form['action']
        username = request.form['username']
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        if action == 'Login':
            # Handle login
            if username in users_db:
                if users_db[username]['password'] == hashed_password:
                    session['username'] = username
                    flash('Logged in successfully!', 'success')
                    return redirect(url_for('features'))
                else:
                    flash('Incorrect password. Please try again.', 'danger')
            else:
                # If user does not exist, ask if they want to sign up
                flash(f'User "{username}" does not exist. Do you want to create an account?', 'info')
                return render_template('login_signup.html', show_signup=True, username=username, password=password)

        elif action == 'Signup':
            # Handle signup
            if username in users_db:
                flash('Username already exists!', 'danger')
            else:
                users_db[username] = {'username': username, 'password': hashed_password}
                flash('Signup successful! Please log in.', 'success')
    
    return render_template('login_signup.html')

@app.route('/welcome_back')
def welcome_back():
    if 'username' in session:
        return render_template('welcome_back.html', username=session['username'])
    return redirect(url_for('login_signup'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('home'))

@app.route('/form', methods=['GET'])
def form():
    return render_template('form.html')

@app.route('/features', methods=['GET'])
def features():
    return render_template('features.html')

@app.route('/handle_form_submission', methods=['GET'])
def handle_form_submission():
    domain = request.args.get('domain')
    other_text = request.args.get('otherText', '')

    # Save form data to the temporary database
    form_selections_db[session.get('username', 'guest')] = {
        'domain': domain,
        'otherText': other_text
    }

    # Redirect to the next page with query parameters
    return redirect(url_for('next_page', domain=domain, otherText=other_text))

@app.route('/next-page')
def next_page():
    domain = request.args.get('domain')
    project_ideas = generate_project_ideas(domain)

    if request.headers.get('Accept') == 'application/json':
        # Return JSON response for AJAX requests
        return jsonify({"ideas": project_ideas})
    
    # Return HTML template for regular requests
    return render_template('next-page.html', domain=domain, project_ideas=project_ideas)

@app.route('/discard-idea', methods=['GET'])
def discard_idea():
    index = request.args.get('index')
    domain = request.args.get('domain')
    
    project_ideas = generate_project_ideas(domain)
    
    if index.isdigit() and int(index) < len(project_ideas):
        new_idea = project_ideas[int(index)]
        return jsonify({
            "newIdea": {
                "index": index,
                "text": new_idea
            }
        })
    else:
        return jsonify({"newIdea": None}), 400

@app.route('/generate-guide', methods=['GET'])
def generate_guide_route():
    selected_idea = request.args.get('idea')
    domain = request.args.get('domain')

    if not selected_idea:
        flash('No project idea selected', 'danger')
        return redirect(url_for('next_page', domain=domain))

    # Use a separate thread to handle synchronous processing
    guide = None

    def worker():
        nonlocal guide
        guide = generate_guide(selected_idea, domain)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()  # Wait for the thread to finish

    # Render the guide on a new page with improved formatting
    return render_template('project_guide.html', project_idea=selected_idea, project_guide=guide)

@app.route('/index')
def index():
    if 'username' not in session:
        flash("You need to be logged in to access Project Visualization.", "warning")
        return redirect(url_for('welcome_back'))  # Redirect to features if not logged in

    username = session['username']
    user_data = users_db.get(username, {})

    problem_statement = request.args.get('problem', None)
    
    if problem_statement:
        checkpoints_data = user_data['problem_statements'].get(problem_statement, [])
        session['problem_statement'] = problem_statement
        session['checkpoints_data'] = checkpoints_data
    else:
        if 'problem_statement' not in session or 'checkpoints_data' not in session:
            # Load user's default problem statement if none are selected
            problem_statement = list(user_data['problem_statements'].keys())[0]
            checkpoints_data = user_data['problem_statements'][problem_statement]
            session['problem_statement'] = problem_statement
            session['checkpoints_data'] = checkpoints_data
        else:
            problem_statement = session['problem_statement']
            checkpoints_data = session['checkpoints_data']

    return render_template('index.html', problem_statement=problem_statement, checkpoints_data=checkpoints_data, problem_statements=user_data.get('problem_statements', {}), skill=user_data.get('skill', 'Beginner'))

@app.route('/checkpoint/<checkpoint_title>')
def checkpoint_page(checkpoint_title):
    checkpoints_data = session.get('checkpoints_data', [])
    selected_checkpoint = next((item for item in checkpoints_data if item['title'] == checkpoint_title), None)
    
    if not selected_checkpoint:
        flash(f"Checkpoint {checkpoint_title} not found.", "danger")
        return redirect(url_for('index'))

    return render_template('checkpoint.html', checkpoint=selected_checkpoint)

if __name__ == '__main__':
    socketio.run(app, debug=True)
