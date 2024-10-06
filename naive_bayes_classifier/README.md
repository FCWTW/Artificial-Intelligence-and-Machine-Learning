# Naive Bayes Classifier Implementation

## Major formula for classifier
![formula](/Image/Bayes_formula.png)

## Different between various Naive Bayes Classifier
> All naive bayes classifier are base on Bayes formula above, but their likelihood function ( P(X|C) ) are different.
>
> Therefore, the types of data they apply to are also different. 

## Likelihood function

* ### Gaussian NB

![G](/Image/Gaussian_likelihood.png)

* ### Bernoulli NB

![B](/Image/Bernoulli_likelihood.png)

## Dataset
* ### iris dataset

* ### [Milk Quality dataset](https://www.kaggle.com/datasets/cpluzshrijayan/milkquality)

---
## Code explanation for Gaussian NB

### 1. prior probabilities

> ![G_PR](/Image/GNB_prior.png)

### 2. likelihood function

> To avoid numerical overflows, logarithmic space calculations are used.
>
> The formula for likelihood function changes to:
>
> ![log_G_like](/Image/log_G_like.png)

### 3. posterior probabilities
>
> Due to logarithmic space calculations, the formula for posterior probabilities (also the major formula at the beginning) changes to:
>
> ![log_post](/Image/log_post.png)
>
> log(prior) is included in the calculation of the likelihood function, so it does not appear.
>
> ![rrrrr](/Image/explain.png)
>
> np.exp is used at the end to obtain the original posterior probabilities.
>
> To facilitate interpretation, np.round is used to retain only the first ten decimal places of the classification results.
>
> ![result](/Image/G_result.png)

### 4. classification result
>
> The predict function uses the logarithmic likelihood function value for classification.
>
> It doesn't affect the classification results.

---
## Code explanation for Bernoulli NB
> 施工中

---
> <details>
>  <summary>Reference</summary>
> 1. <a href="https://www.learncodewithmike.com/2020/11/python-pandas-dataframe-tutorial.html">DataFrame處理教學</a><br>
> 2. <a href="https://stackoverflow.com/questions/57817758/badzipfile-file-is-not-a-zip-file-error-popped-up-all-of-a-sudden">different pd.read_* method</a><br>
> 3. <a href="https://blog.demir.io/hands-on-numpyro-a-practitioners-guide-to-building-bernoulli-naive-bayes-with-confidence-abc94f11cf9a">Guide to Building Bernoulli Naive Bayes</a><br>
> 4. <a href="https://roger010620.medium.com/%E8%B2%9D%E6%B0%8F%E5%88%86%E9%A1%9E%E5%99%A8-naive-bayes-classifier-%E5%90%ABpython%E5%AF%A6%E4%BD%9C-66701688db02">貝氏分類器(Naive Bayes Classifier)(含python實作)</a><br>
> 5. <a href="https://github.com/amallia/GaussianNB/blob/master/gaussianNB.py">GaussianNB (GitHub)</a><br>
> 6. <a href="https://www.kaggle.com/code/prashant111/naive-bayes-classifier-in-python#12.-Model-training-">Naive Bayes Classifier in Python</a><br>
> </details>