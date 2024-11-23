# Text Classification using BERT

### Overview
This project utilizes **BERT (Bidirectional Encoder Representations from Transformers)** to classify BBC news articles into five predefined categories:  
- **Business**  
- **Entertainment**  
- **Politics**  
- **Sport**  
- **Technology**  

We leverage fine-tuning of a pre-trained BERT model to achieve high accuracy in text classification.

---

### Objective
The main goal of this project is to develop a robust machine-learning pipeline capable of generalizing well to unseen text data, providing high classification accuracy across different categories.

---

### Dataset
- **Name**: BBC News Articles Dataset  
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category?select=bbc-text.csv)  
- **Details**: This dataset contains labeled news articles, categorized into five classes.  

---

### Methodology
1. **Preprocessing**  
   - Tokenization using `bert-base-cased`.
   - Padding and truncating text sequences to a maximum length of 512 tokens.
   - Mapping labels to numerical representations.  
     Example: `business: 0`, `entertainment: 1`, etc.

2. **Model Architecture**  
   - Pre-trained BERT backbone for contextual embeddings.  
   - Fine-tuned layers for classification with dropout and batch normalization.  
   - Mean pooling across all token embeddings for robust sentence representations.

3. **Training Pipeline**  
   - **Loss Function**: Cross-entropy.  
   - **Optimizer**: AdamW with linear learning rate decay.  
   - **Gradient Accumulation**: For memory-efficient training on GPUs.  
   - Validation after each epoch to monitor performance.

4. **Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-Score.  
   - Classification reports and confusion matrices for detailed insights.

---

### Results
The BERT model achieved outstanding performance, significantly outperforming traditional machine learning models such as Random Forest and Logistic Regression:  

| Model                  | Test Accuracy | Precision | Recall | F1-Score |
|------------------------|---------------|-----------|--------|----------|
| **BERT Model**         | 90%           | 91%       | 90%    | 90%      |
| Random Forest          | 81%           | 82%       | 81%    | 81%      |
| Logistic Regression    | 78%           | 78%       | 78%    | 78%      |
| Multinomial Naive Bayes| 78%           | 79%       | 78%    | 78%      |

---

### How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/chaitanya-reddy-13/TEXT_Classification_Using_BERT.git
