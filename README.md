# AI Project Guide Tool
 
## Overview

The Project Guide Tool is a web-based platform that helps users generate problem statements based on selected categories and ensures fair task distribution within teams. It utilizes locally hosted LLMs to suggest project ideas and an AI-powered workload distribution system to assign tasks based on team skills and project requirements.

## Features

Problem Statement Generation: Uses Llama 3.1B, Mistral, and Qwen via Ollama to generate relevant project ideas.

AI-Based Workload Distribution: A Random Forest model analyzes team skills and assigns tasks fairly.

Intuitive Web UI: A seamless user experience for selecting categories and managing teams.

Efficient Local Execution: Runs all models locally for low latency and privacy.

## Tech Stack

LLMs: Llama 3.1B, Mistral, Qwen (via Ollama)

Machine Learning: Random Forest for task allocation

Backend: Python API (Flask/FastAPI)

Frontend: React.js

Database: MongoDB (for storing project ideas and team information)

## Usage

Select a project category to generate problem statements.

Enter team members' skills to receive an optimized task distribution.

View recommended project ideas and assigned roles.

## Future Enhancements

Automated Feedback System: AI-generated insights on project feasibility.

Progress Tracking: Integration with task management tools.

GitHub Integration: Auto-generate repositories with project templates.
