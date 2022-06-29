# simpleTextCLEF

This software was developed for the CLEF 2022 Text Simplification task.

Our work uses the transfer learning capabilities of the
T5 pre-trained language model, adding a method to control specific simplification features. We
present a new feature based on masked tokens prediction (Language Model Fill-Mask) to
control the lexical complexity of the text generation process. The results obtained with the
SARI metric are at the same level as previous work in other domains for sentence
simplification.

Steps to replicate the results: 

1. Clone this repository
2. Install dependencies:
<pre><code>pip install -r requirements.txt</code></pre>
3. For training purpose:

Select hyperparameters in T5_train.py 
<pre><code>python scripts/T5_train.py</code></pre>
4. Optimization:

Select experiment_id, dataset and trials in optimization.py
<pre><code>python scripts/optimization.py</code></pre>
5. For test purpose:

Select experiment_id and dataset in T5_evaluate.py
<pre><code>python scripts/T5_evaluate.py</code></pre>

Same for larger version. Be carefull with memory issues. 



# Data

Download the dataset from https://simpletext-project.com/2022/clef/en/tasks.