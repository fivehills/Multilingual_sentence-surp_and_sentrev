# Computational Sentence-level Metrics README

This repository contains code and data supporting the paper *"Computational Sentence-level Metrics of Reading Speed and its Ramifications for Sentence Comprehension"* by Kun Sun and Rong Wang, published in *Cognitive Science*.

## Contents

- **Code**: Python scripts for computing sentence-level metrics proposed in the paper.
- **Data**: References to the MECO eye-tracking dataset used in the study.

## Data

The eye-tracking data used in this study is the Multilingual Eye-movement Corpus (MECO), available at [OSF](https://osf.io/3527a/files/osfstorage).

### Computing Reading Speed
**Reading speed** measures how quickly a person reads a sentence, expressed in words per minute (wpm). Note that the `rate` metric in the MECO dataset differs from reading speed and should not be used interchangeably. To compute reading speed using the MECO dataset:

1. **Count the number of words** in the sentence (available as `sent.nwords`).
2. **Measure the total fixation duration** on the sentence (in seconds, from eye-tracking data).
3. **Convert the total durations of all words within one sentence to minutes** by dividing by 60.
4. **Divide the number of words by the time in minutes**.

**Formula:**

```math
Reading speed (wpm) = sent.nwords / (total fixation durations in one sentence / 60)
```

This calculation provides the reading speed in words per minute (wpm) for each sentence.

## Scripts

### comput_sentrev.py
This Python script computes **sentence relevance** (\( sentrev \)), a metric introduced in the paper. It calculates the semantic relationship between a target sentence and its surrounding context using a weighted sum of cosine similarities, based on a four-sentence sliding window.

### comput_sentsurp.py
This Python script computes **sentence surprisal** (\( sentsurp \)), another metric proposed in the paper. It calculates the surprisal of a sentence given its preceding context, using the chain rule to estimate the joint probability of the sentence’s tokens.

## Usage

To run the scripts, ensure you have the following dependencies installed:
- Python 3.8+
- Libraries: `nltk`, `torch`, `transformers`, `pandas`, `numpy`
- Install dependencies using:

```
pip install nltk torch transformers pandas numpy
```
  
1. Place the input text files (as referenced in the scripts) in the working directory.
2. Run the scripts:
```bash
   python comput_sentrev.py
   python comput_sentsurp.py
```
3. Outputs are saved as CSV files containing the computed metrics.

## Citation

Please cite the following paper if you use this code or data:

Sun, K., & Wang, R. (2025). Computational Sentence-level Metrics of Reading Speed and its Ramifications for Sentence Comprehension. *Cognitive Science*. 

## Contact

For questions or issues, contact the authors via the details provided in the paper.
sharpksun@hotmail.com
