# Simple Multimodal Chatbot with Memory

A lightweight, CLI-based conversational agent that processes text and image inputs, remembers past interactions, and responds contextually. Built with Python, spaCy (text processing), MobileNet (image classification), and a simple memory system, it runs locally on a CPU.

## Overview
This project extends my [Simple Multimodal Agent](https://github.com/sakhileln/multimodal-agent) by adding conversational memory. It takes text prompts (e.g., "What is this?") and image files (e.g., "dog.jpg") via CLI, processes them, and maintains context across interactions. For example:
- Input 1: `--text "What is this?" --image "dog.jpg"` → "This is a dog."
- Input 2: `--text "Is it big?"` → "The dog doesn’t look very big."

The focus is on learning conversational AI, context management, and memory integration while keeping it simple and local.

## Features
- Text intent extraction (e.g., "describe", "classify") using spaCy.
- Image classification using pre-trained MobileNet.
- Memory system to track conversation history (inputs and outputs).
- Context-aware responses based on prior interactions.
- CLI interface with `argparse`.

## Requirements
- Python 3.8+
- Libraries:
  - `spacy==3.7.2` (with `en_core_web_sm` model)
  - `opencv-python==4.9.0.80`
  - `tensorflow==2.15.0`
  - `numpy==1.26.4`
- No new dependencies beyond the above.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sakhileln/multimodal-chatbot-with-memory.git
   cd multimodal-chatbot-with-memory
   ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the spaCy model:
    ```bash
    python -m spacy download en_core_web_sm
    ```
## Usage
- Run the chatbot via CLI:
    ```bash
    python chatbot.py --text "What is this?" --image "path/to/dog.jpg"
    ```
- Follow up with a context-aware prompt:
    ```bash
    python chatbot.py --text "Is it big?"
    ```
## Project Structure
- `chatbot.py`: Main script with CLI, memory, and response logic.
- `memory.py`: Memory system to store conversation history.
- `text_processor.py`: Text intent extraction with spaCy.
- `image_processor.py`: Image classification with MobileNet.
- `requirements.txt`: Dependencies list.

## Goals
- Deepen understanding of conversational AI and context management.
- Build a reusable, modular codebase for future experiments.

## License
This project is licensed under the GPL v3.0 License. See the [LICENSE](LICENSE) file for details.

## Contact

- Sakhile III  
- [LinkedIn Profile](https://www.linkedin.com/in/sakhile-ndlazi)
- [GitHub Profile](https://github.com/sakhileln)
