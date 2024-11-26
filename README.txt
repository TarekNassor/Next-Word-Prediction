# Next Word Prediction Model

This project demonstrates a **Next Word Prediction** system using a deep learning approach. The model predicts the next word(s) in a given sequence of text based on a dataset derived from the book *The Modern Prometheus*.

## Features
- Tokenizes text and creates sequences for training.
- Uses a **LSTM-based neural network** for next word prediction.
- Implements **ROUGE evaluation** to assess the quality of predictions.
- Allows users to input a seed text and generate the next set of words.

## Dataset
The model uses *The Modern Prometheus* as the training dataset. The text is tokenized, and n-gram sequences are generated for model training.

## Model Architecture
- **Embedding Layer**: Converts words into dense vector representations.
- **LSTM Layer**: Processes sequential data to learn context.
- **Dense Layer**: Outputs probabilities for each word in the vocabulary.

## Usage

### 1. Train the Model
The model is trained using sequences generated from the text. It uses categorical cross-entropy as the loss function and Adam optimizer for training.

### 2. Predict Next Words
The `predict_next_n_words` function takes a seed text and predicts a specified number of next words:
```python
seed_text = "The master is"
next_words = 6
generated_text = predict_next_n_words(seed_text, next_words)
print(generated_text)
```

### 3. Evaluate the Model
The model's predictions are evaluated using the **ROUGE metric** by comparing generated text with reference sentences:
```python
scores = rouge.get_scores(predictions, reference_sentences, avg=True)
print(scores)
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/next-word-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements
- Python 3.7+
- TensorFlow
- NumPy
- ROUGE

## Example Output
- **Seed Text**: `The master is`
- **Generated Text**: `The master is not a man who`

### Evaluation Scores
ROUGE-L: 0.47 (example score, varies based on dataset and training)

## Future Enhancements
- Support for larger datasets and pre-trained embeddings.
- Add a web interface for interactive usage.
- Experiment with Transformer-based models like GPT or BERT.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The book *The Modern Prometheus* for providing the dataset.
- TensorFlow and the Keras library for enabling deep learning workflows.

---

Feel free to explore and contribute! 😊