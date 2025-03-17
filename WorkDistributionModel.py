import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import spacy
import re

# Load spaCy model for word embeddings
nlp = spacy.load('en_core_web_md')

# Expanded task descriptions and their relevant skills
data = {
    'Task Description': [
        'Create a new website homepage',
        'Implement a machine learning model',
        'Set up a firewall for network security',
        'Design a responsive UI for a mobile app',
        'Write an ETL pipeline for data processing',
        'Conduct a security audit for the web application',
        'Optimize SQL queries for performance',
        'Build a REST API using Flask',
        'Train a neural network for image classification',
        'Create a deployment pipeline for CI/CD',
        'Develop a chatbot using NLP',
        'Design an interactive dashboard',
        'Set up cloud infrastructure using AWS',
        'Implement user authentication in a web app',
        'Perform vulnerability scanning on the network',
        'Create data visualizations with Python',
        'Refactor legacy code for better maintainability',
        'Design a microservices architecture',
        'Create a recommendation system',
        'Build and manage a database schema',
        'Develop a mobile application with React Native',
        'Conduct A/B testing for feature improvements',
        'Create a machine learning pipeline with scikit-learn',
        'Implement a containerization strategy using Docker',
        'Secure API endpoints with OAuth',
        'Design a scalable cloud architecture',
        'Implement data encryption and decryption',
        'Create automated reports with data analysis',
        'Build a high-performance web server',
        'Set up and manage Kubernetes clusters',
        'Develop an AI-based fraud detection system',
        'Create a multi-tier application architecture',
        'Design a user-friendly admin panel',
        'Implement data integrity checks',
        'Conduct threat modeling for a new application'
    ],
    'Skill': [
        'Web Development',
        'Machine Learning',
        'Cybersecurity',
        'UI/UX Design',
        'Data Engineering',
        'Cybersecurity',
        'Database Management',
        'Web Development',
        'Machine Learning',
        'DevOps',
        'Machine Learning',
        'UI/UX Design',
        'DevOps',
        'Web Development',
        'Cybersecurity',
        'Data Science',
        'Software Engineering',
        'DevOps',
        'Machine Learning',
        'Database Management',
        'Web Development',
        'UI/UX Design',
        'Machine Learning',
        'DevOps',
        'Cybersecurity',
        'Cloud Engineering',
        'Cybersecurity',
        'Data Science',
        'Web Development',
        'DevOps',
        'Machine Learning',
        'Software Engineering',
        'UI/UX Design',
        'Database Management',
        'Cybersecurity'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode skill labels into numeric values
label_encoder = LabelEncoder()
df['Skill_Label'] = label_encoder.fit_transform(df['Skill'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Task Description'], df['Skill_Label'], test_size=0.2, random_state=42)

# Use TfidfVectorizer with ngram_range and min_df to improve text representation
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
model = make_pipeline(vectorizer, RandomForestClassifier(n_estimators=100, random_state=42))

# Train the model
model.fit(X_train, y_train)

# Check the accuracy on the test set
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Function to predict the skill based on task description
def predict_skill(task_description):
    skill_label = model.predict([task_description])[0]
    return label_encoder.inverse_transform([skill_label])[0]

# Fallback rule-based skill prediction for very generic terms
def fallback_skill_prediction(task_description):
    task_description = task_description.lower()
    
    if re.search(r'\bapp\b|\bmobile\b', task_description):
        return 'UI/UX Design'
    elif re.search(r'\bsoftware\b', task_description):
        return 'Software Engineering'
    elif re.search(r'\bnetwork\b|\bsecurity\b', task_description):
        return 'Cybersecurity'
    elif re.search(r'\bdatabase\b|\bqueries\b', task_description):
        return 'Database Management'
    elif re.search(r'\bcloud\b|\baws\b|\bdevops\b|\bcontainer\b', task_description):
        return 'DevOps'
    elif re.search(r'\bai\b|\bml\b|\bneural network\b|\bmachine learning\b', task_description):
        return 'Machine Learning'
    else:
        # Default back to machine learning model prediction if no keyword match
        return predict_skill(task_description)

# Function to find the most similar skill if the exact skill is missing
def find_similar_skill(skill, known_skills):
    skill_vector = nlp(skill)
    similar_skills = {known_skill: skill_vector.similarity(nlp(known_skill)) for known_skill in known_skills}
    return max(similar_skills, key=similar_skills.get)

# Define the team and their skills
team_data = {
    'Person A': {'Web Development': 'Intermediate', 'Machine Learning': 'Noob', 'Cybersecurity': 'Pro', 'UI/UX Design': 'Intermediate', 'Data Engineering': 'Intermediate', 'Database Management': 'Noob', 'DevOps': 'Noob', 'Data Science': 'Intermediate', 'Software Engineering': 'Noob', 'Cloud Engineering': 'Noob'},
    'Person B': {'Web Development': 'Pro', 'Machine Learning': 'Intermediate', 'Cybersecurity': 'Noob', 'UI/UX Design': 'Noob', 'Data Engineering': 'Intermediate', 'Database Management': 'Pro', 'DevOps': 'Pro', 'Data Science': 'Noob', 'Software Engineering': 'Intermediate', 'Cloud Engineering': 'Pro'},
    'Person C': {'Web Development': 'Noob', 'Machine Learning': 'Pro', 'Cybersecurity': 'Intermediate', 'UI/UX Design': 'Intermediate', 'Data Engineering': 'Noob', 'Database Management': 'Pro', 'DevOps': 'Noob', 'Data Science': 'Pro', 'Software Engineering': 'Pro', 'Cloud Engineering': 'Intermediate'},
    'Person D': {'Web Development': 'Intermediate', 'Machine Learning': 'Noob', 'Cybersecurity': 'Intermediate', 'UI/UX Design': 'Pro', 'Data Engineering': 'Noob', 'Database Management': 'Intermediate', 'DevOps': 'Pro', 'Data Science': 'Intermediate', 'Software Engineering': 'Intermediate', 'Cloud Engineering': 'Noob'}
}

# Convert the team data to a DataFrame
team_df = pd.DataFrame.from_dict(team_data, orient='index')

# Define proficiency levels and their weights
proficiency_weights = {'Noob': 1, 'Intermediate': 2, 'Pro': 3}

# Convert skill levels to numeric weights
for skill in team_df.columns:
    team_df[skill] = team_df[skill].map(proficiency_weights)

# Input tasks dynamically with complexity level
n_tasks = int(input("Enter the number of tasks: "))
tasks = []
for i in range(n_tasks):
    task_name = input(f"Enter description for Task {i+1}: ")
    complexity = int(input(f"Enter complexity level for Task {i+1} (1-5): "))
    
    # Use fallback method to predict skill
    predicted_skill = fallback_skill_prediction(task_name)
    print(f"Predicted skill for '{task_name}' is: {predicted_skill}")
    
    # Check if all team members have the predicted skill, else find similar skill
    if predicted_skill not in team_df.columns:
        similar_skill = find_similar_skill(predicted_skill, team_df.columns)
        print(f"No exact skill match found. Using similar skill: {similar_skill}")
        predicted_skill = similar_skill
    
    tasks.append((task_name, predicted_skill, complexity))

# Initialize task assignments
assignments = {task[0]: {} for task in tasks}

# Distribute tasks based on predicted skill proficiency and task complexity
for task, skill, complexity in tasks:
    skill_proficiency = team_df[skill]
    
    # Normalize skill proficiency to percentage for the task considering complexity
    total_skill = skill_proficiency.sum()
    for person in team_df.index:
        # Multiply by complexity to adjust workload
        assignments[task][person] = (team_df.loc[person, skill] / total_skill) * 100 * complexity

    # Normalize to ensure the percentages sum to 100%
    total_percentage = sum(assignments[task].values())
    assignments[task] = {person: (percentage / total_percentage) * 100 for person, percentage in assignments[task].items()}

# Print the work division
print("\nWork Division:")
for task, distribution in assignments.items():
    print(f"\n{task}:")
    for person, percentage in distribution.items():
        print(f"  {person}: {percentage:.2f}%")
