# Consumer_Complaints

Fine-tuning DistilBERT on complaints.csv data

This Repository contains eight branches:

**main branch** : Contains information about the Data source and the requirements

**Experiment1** : Fine-tuning DistilBERT on a sample data of 25619 entries from the complaints.csv data, which was pre-processed using spaCy library.

**Experiment3** : Fine-tuning DistilBERT on a sample data of 37487 entries from the complaints.csv data, which was pre-processed using spaCy library.

**Experiment5** : Fine-tuning DistilBERT on a sample data of 42736 entries from the complaints.csv data, which was pre-processed using spaCy and nltk libraries.

**Experiment6** : Fine-tuning DistilBERT on a balanced sample of 60000 entries that has 10000 entries of each of the 'product' counts and was pre-processed using Gensim library for 5 epochs. The stop-words are removed using Gensim library.

**Experiment7** : Fine-tuning DistilBERT on a balanced sample of 60000 entries that has 10000 entries of each of the 'product' counts and was pre-processed using Gensim library for 10 epochs. The stop-words are not removed.

**Experiment8** : Fine-tuning DistilBERT on a balanced sample of 60000 entries that has 10000 entries of each of the 'product' counts and was pre-processed using Gensim library for 10 epochs. The stop-words are removed using Gensim library.

**Final** : Fine-tuning DistilBERT on a balanced sample of 60000 entries that has 10000 entries of each of the 'product' counts and was pre-processed using Gensim library for 5 epochs. The stop-words are not removed. The model's performance on the Test Dataset is the best, among the other experiments.
