# Apriori Algorithm Implementation

Apriori is an algorithm for association rule mining that aims to discover hidden associations between items from transaction data.

This algorithm is based on a simple principle: if an itemset is frequent, then all its subsets must also be frequent.

![formula](/Image/apriori_formula.png)

* Support: The frequency of an item set appearing in the data set is used to measure whether an item set is frequent enough.

* Confidence: The probability that item set Y also appears when item set X is given.

* Lift: A measure of the strength of the association between X and Y.

---
![flow](/Image/apriori_flow.png)

> <details>
>  <summary>Reference</summary>
> 1. <a href="https://www.kaggle.com/code/parisanahmadi/how-to-solve-the-apriori-algorithm-in-a-simple-way">How to solve the Apriori algorithm in a simple way</a><br>
> 2. <a href="https://www.softwaretestinghelp.com/apriori-algorithm/">Apriori Algorithm in Data Mining: Implementation With Examples</a><br>
> </details>