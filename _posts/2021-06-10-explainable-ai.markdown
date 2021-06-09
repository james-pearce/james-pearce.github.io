---
layout: post
title:  "Why we do—and don't—need explainable AI"
date:   2021-06-09 09:00:00 +1100
categories: machine-learning XAI SHAP
---

_Why explainable AI (known as XAI) is becoming a must-have component of data science and why we may not have come as far as we think._

**A black-box model is no longer good enough for your data scientists, your business or your customers.**

## A data science history perspective

Back in the old days (really just a few years ago for some) building models for classification was simple. [Generalised linear models](https://en.wikipedia.org/wiki/Generalized_linear_model) (GLMs), and in particular, [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) models were the only algorithms we used.

And this is the _real_, statistical version of logistic regression as you would find in the [Python](https://www.python.org) library [`statsmodels`](https://www.statsmodels.org/stable/glm.html), rather than the [`scikit-learn`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) version. In the statistical version, you get neat statistical parameter information including [coefficient estimates](https://www.sciencedirect.com/topics/engineering/regression-coefficient), [standard errors](https://en.wikipedia.org/wiki/Standard_error) and estimates of fit such as [deviance](https://en.wikipedia.org/wiki/Deviance_(statistics)) or [log-likelihood](https://online.stat.psu.edu/stat504/lesson/1/1.5) estimates.

Our models contained a few, essential [features](https://en.wikipedia.org/wiki/Feature_%28machine_learning%29). Those features we did use were handcrafted to be meaningful, stable and predictable.

## The new kid in town

Then things changed. They got more sophisticated and more complex (possibly two words meaning the same thing). We were given a choice of many different algorithms, such as [neural networks](https://scikit-learn.org/stable/modules/neural_networks_supervised.html), [random forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [gradient boosting machines](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), [support vector machines](https://scikit-learn.org/stable/modules/svm.html) and [ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html). We could now use hundreds — or even thousands — of features. In many cases our algorithms would select the best features and even pre-process them, meaning we no longer had to hand-craft them. We started to use a whole bunch of techniques that, when combined, would wring every drop of predictive power from our models and data.

The mantra was sometimes heard that as long as the model predicted well, why should we care about how the model arrived at its predictions?

And our debugging and diagnostic techniques evolved accordingly. They did not try to explain _why_ the model produced a prediction. Rather, [they quantified the extent to which the algorithm would generalise to a set of unseen data](https://www.datasciencecentral.com/profiles/blogs/7-important-model-evaluation-error-metrics-everyone-should-know).

## Constraints can be good for creativity

Admittedly, in the scenario I presented in the old days, a lot of these constraints were thrust upon us. Our software did not support advanced algorithms. The lack of computational power made thousands of features impractical and time-consuming. Putting models into operational systems was hard; it often required hand-coding the algorithm into a mainframe (and yes, this is something I have done _in the last decade_). Altogether this pushed the data scientist towards simpler models and implementations that were easier to test.

So with all these constraints in place, the models and algorithms developed with understanding and interpretability first and foremost. Contributing to why this needed to be so was what the models were used for: things like determining whether to grant an applicant a credit card, for example. It _just felt safer_ that humans could understand these models, regardless of the checks and balances in the process. We were risking hundreds and thousands of dollars if we got it wrong.

But for now, enough _blah, blah, blah._ I will show you what I mean using the classic [_Titanic_ data set](https://www.kaggle.com/hesh97/titanicdataset-traincsv).

---

> **Spoiler alert**
>
> The ship hits an iceberg and sinks.
---

We want to build a model that will predict who would survive on the _Titanic_.[^1]

[^1]: A prediction that will be really helpful if you want to book safe passage on _Titanic II_’s maiden voyage.

![modelling](/assets/modelling.svg)



To build a model, I use a logistic regression in [R](http://cran.r-project.org) using the [`glm`](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/glm) package. You can see my code [here](https://github.com/james-pearce/titanic-xai).

Comparing our predictions with what transpired, we get the following misclassification table (or _confusion matrix_).

|                     |              Actual |              |
| ------------------- | ------------------: | -----------: |
| **Predicted**       | **did not survive** | **survived** |
| **did not survive** |                 480 |           84 |
| **survived**        |                  69 |          258 |

If you add up the numbers, you can see this model gives an accuracy on the data set it was trained on of 82%. This is the number of times we made a prediction that matched with what actually happened.

Great! Our model is somewhat accurate. But it does not help me in my quest for surviving when _Titanic II_ hits an iceberg. For that we have to look into the internals of the fitted model and do some simple maths.

### Analysis of deviance

First, I look at the [analysis of deviance](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/anova.glm) table. This tells me that all the variables except `Fare`, `Parch` and `Embarked` contribute to reducing the variation or error. It also tells me the bulk of predictive power lives in two variables: `Sex` and `Pclass`.



|          |   Df | Deviance | Resid. Df | Resid. Dev | $\Pr(>\chi^2)$ |
| -------- | ---: | -------: | --------: | ---------: | -------------: |
| NULL     |   NA |       NA |       890 |     1186.7 |             NA |
| Pclass   |    2 |    103.5 |       888 |     1083.1 |           0.00 |
| Sex      |    1 |    256.2 |       887 |      826.9 |           0.00 |
| age_band |    7 |     25.6 |       880 |      801.2 |           0.00 |
| SibSp    |    2 |     16.7 |       878 |      784.6 |           0.00 |
| Parch    |    2 |      3.9 |       876 |      780.7 |           0.14 |
| Fare     |    4 |      1.4 |       872 |      779.2 |           0.84 |
| Cabin    |    7 |     15.7 |       865 |      763.5 |           0.03 |
| Embarked |    2 |      5.0 |       863 |      758.5 |           0.08 |

### Table of coefficients

Fitting a generalised linear model also gives us a table of coefficients. We can use this to see which values lead to greater or lesser chances of survival.

```R
## Call:
## glm(formula = Survived ~ Pclass + Sex + age_band + SibSp + Parch +
##     Fare + Cabin + Embarked, family = binomial, data = titanic_df)
##
## Deviance Residuals:
##     Min       1Q   Median       3Q      Max
## -2.5514  -0.6254  -0.3818   0.5720   2.5841
##
## Coefficients:
##                         Estimate Std. Error z value Pr(>|z|)
## (Intercept)              0.10832    0.77843   0.139 0.889334
## Pclass1                  1.39381    0.52904   2.635 0.008424 **
## Pclass2                  1.05813    0.30546   3.464 0.000532 ***
## Sexfemale                2.73557    0.21185  12.913  < 2e-16 ***
## age_band 2. 18 to 24    -1.36944    0.37236  -3.678 0.000235 ***
## age_band 3. 25 to 34    -1.08831    0.35717  -3.047 0.002311 **
## age_band 4. 35 to 44    -1.53207    0.39933  -3.837 0.000125 ***
## age_band 5. 45 to 54    -2.01319    0.45186  -4.455 8.38e-06 ***
## age_band 6. 55 to 64    -2.21546    0.60183  -3.681 0.000232 ***
## age_band 7. 65 and over -3.35952    1.15066  -2.920 0.003504 **
## age_band 8. Other       -1.37755    0.37881  -3.636 0.000276 ***
## SibSp1                  -0.02670    0.25043  -0.107 0.915082
## SibSp2+                 -1.30675    0.42860  -3.049 0.002297 **
## Parch1                   0.17985    0.31417   0.572 0.567005
## Parch2+                 -0.42993    0.35948  -1.196 0.231706
## Fare 2. ( 10,  20]       0.09089    0.32287   0.282 0.778322
## Fare 3. ( 20,  30]       0.06864    0.39740   0.173 0.862865
## Fare 4. ( 30,  40]       0.09779    0.49297   0.198 0.842758
## Fare 5. ( 40,    ]       0.34598    0.50561   0.684 0.493798
## CabinB                   0.06157    0.71413   0.086 0.931292
## CabinC                  -0.43524    0.66889  -0.651 0.515244
## CabinD                   0.35476    0.75164   0.472 0.636945
## CabinE                   0.88119    0.77044   1.144 0.252730
## CabinF                   0.39872    1.01748   0.392 0.695157
## CabinG/T                -1.94008    1.28924  -1.505 0.132368
## CabinNot given          -0.84486    0.68418  -1.235 0.216887
## EmbarkedQ                0.04961    0.40881   0.121 0.903410
## EmbarkedS               -0.47637    0.25177  -1.892 0.058481 .
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
##
## (Dispersion parameter for binomial family taken to be 1)
##
##     Null deviance: 1186.66  on 890  degrees of freedom
## Residual deviance:  758.52  on 863  degrees of freedom
## AIC: 814.52
##
## Number of Fisher Scoring iterations: 5
```

This tells us:

* `Pclass` of 3 is associated with non-survival;
* `Sex` of ‘female’ is associated with survival;
* Children (`Age` less than 18) are more likely to survive; and
* `Cabin` designations beginning with `G` are unlikely to survive,

as well as some other, less predictive, nuances.

### Explaining predictions

Now we have managed to understand what is happening in our Titanic model. The next step is to _understand_ why the model makes individual predictions.

Let’s start by looking at the prediction of the passenger who was predicted to me _most_ likely to survive. Here are the details of that passenger, who has been predicted as 99.2% likely to survive.

| Pclass<fct> | Sex<fct> | age_band<fct> | SibSp<fct> | Parch<fct> | Fare<fct>  | Cabin<fct> | Embarked<fct> |
| :---------- | :------- | :------------ | :--------- | :--------- | :--------- | :--------- | :------------ |
| 1           | female   | 1. Under 18   | 0          | 1          | 5. ( 40, ] | B          | C             |

All we need to do is a bit of mathematics to understand why this was predicted.

```R
0.10831603 +
  1.39380902 + # Pclass 1
  2.73556786 + # Sex female
  0 + # age_band 1. Under 18
  0 + # SibSp 0
  0.17985123 + # Parch 1
  0.34597961 + # Fare 40+
  0.06157172 + # Cabin B
  0  # Embarked C
```

So, it is because she is female, `Pclass` 1, under 18 and paid fare of over 40, along with some other details.

Now let’s look at the passenger predicted to be least likely to be a survivor. Here are the details of that passenger, predicted with a 1.0% chance of surviving.

| Pclass<fct> | Sex<fct> | age_band<fct>  | SibSp<fct> | Parch<fct> | Fare<fct>   | Cabin<fct> | Embarked<fct> |
| :---------- | :------- | :------------- | :--------- | :--------- | :---------- | :--------- | :------------ |
| 3           | male     | 7. 65 and over | 0          | 0          | 1. ( 0, 10] | Not given  | S             |

Again, we can see why this prediction was made with a bit of maths.

```R
0.10831603 +
  0 + # Pclass 3
  0 + # Sex male
  -3.35951828 + # age_band 7. 65 and over
  0 + # SibSp 0
  0 + # Parch 0
  0 + # 1. (0, 10]
  -0.84485644 + # Cabin Not given
  -0.47636764  # Embarked S
```

It is because

* the passenger was male;
* he was aged 65 or over;
* he was in `Pclass` 3; and
* paid a low fare,

as well as other attributes that contribute to the low prediction.

### Interpretable and explainable

It is interesting to note that the old-school approaches grounded in statistics gave us models that were _interpretable_ and produced predictions that were _explainable_.

---

## Modern machine learning

Move forward to today. Machine learning is moving to the mainstream and there are a plethora of tools we can use to build a predictive mode. We are no longer confined to ‘just’ the regression family (although there is a new, improved machine-learning version of regression, too).

The data scientist can choose from a dazzling array of algorithms. As an example, [`lazypredict`](https://lazypredict.readthedocs.io/en/latest/) will automatically fit [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html), [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html), [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression), [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier#sklearn.ensemble.RandomForestClassifier), [`GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html?highlight=gradientboostingclassifier#sklearn.ensemble.GradientBoostingClassifier), [`GaussianNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html?highlight=gaussiannb#sklearn.naive_bayes.GaussianNB) and another _twenty-five more!_

```python
| Model                          |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time Taken |
|:-------------------------------|-----------:|--------------------:|----------:|-----------:|-------------:|
| LinearSVC                      |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0150008 |
| SGDClassifier                  |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0109992 |
| MLPClassifier                  |   0.985965 |            0.986904 |  0.986904 |   0.985994 |    0.426     |
| Perceptron                     |   0.985965 |            0.984797 |  0.984797 |   0.985965 |    0.0120046 |
| LogisticRegression             |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.0200036 |
| LogisticRegressionCV           |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.262997  |
| SVC                            |   0.982456 |            0.979942 |  0.979942 |   0.982437 |    0.0140011 |
| CalibratedClassifierCV         |   0.982456 |            0.975728 |  0.975728 |   0.982357 |    0.0350015 |
| PassiveAggressiveClassifier    |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0130005 |
| LabelPropagation               |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0429988 |
| LabelSpreading                 |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0310006 |
| RandomForestClassifier         |   0.97193  |            0.969594 |  0.969594 |   0.97193  |    0.033     |
| GradientBoostingClassifier     |   0.97193  |            0.967486 |  0.967486 |   0.971869 |    0.166998  |
| QuadraticDiscriminantAnalysis  |   0.964912 |            0.966206 |  0.966206 |   0.965052 |    0.0119994 |
| HistGradientBoostingClassifier |   0.968421 |            0.964739 |  0.964739 |   0.968387 |    0.682003  |
| RidgeClassifierCV              |   0.97193  |            0.963272 |  0.963272 |   0.971736 |    0.0130029 |
| RidgeClassifier                |   0.968421 |            0.960525 |  0.960525 |   0.968242 |    0.0119977 |
| AdaBoostClassifier             |   0.961404 |            0.959245 |  0.959245 |   0.961444 |    0.204998  |
| ExtraTreesClassifier           |   0.961404 |            0.957138 |  0.957138 |   0.961362 |    0.0270066 |
| KNeighborsClassifier           |   0.961404 |            0.95503  |  0.95503  |   0.961276 |    0.0560005 |
| BaggingClassifier              |   0.947368 |            0.954577 |  0.954577 |   0.947882 |    0.0559971 |
| BernoulliNB                    |   0.950877 |            0.951003 |  0.951003 |   0.951072 |    0.0169988 |
| LinearDiscriminantAnalysis     |   0.961404 |            0.950816 |  0.950816 |   0.961089 |    0.0199995 |
| GaussianNB                     |   0.954386 |            0.949536 |  0.949536 |   0.954337 |    0.0139935 |
| NuSVC                          |   0.954386 |            0.943215 |  0.943215 |   0.954014 |    0.019989  |
| DecisionTreeClassifier         |   0.936842 |            0.933693 |  0.933693 |   0.936971 |    0.0170023 |
| NearestCentroid                |   0.947368 |            0.933506 |  0.933506 |   0.946801 |    0.0160074 |
| ExtraTreeClassifier            |   0.922807 |            0.912168 |  0.912168 |   0.922462 |    0.0109999 |
| CheckingClassifier             |   0.361404 |            0.5      |  0.5      |   0.191879 |    0.0170043 |
| DummyClassifier                |   0.512281 |            0.489598 |  0.489598 |   0.518924 |    0.0119965 |
```

**_A list of classifiers from `lazypredict`’s documentation._**

What a long way we have come …



### … Or have we?

My original analytical training was as a statistician. This means I have a bias, so please keep that in mind.

![A statistician quotation](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.SDJJ0y7zXX5HVwuZXBIaQwHaDf%26pid%3DApi&f=1)

With the classical, statistical techniques the focus was on _inference_; the new machine learning techniques focus on _prediction_. For a while there was a sentiment—and indeed, a junior aspiring data scientist said this to me—that so long as a model predicts well, it does not matter what happens inside the algorithm’s black box.

_The end justifies the means._

Of course, now we know that it *does* matter what happens; we do need to be able to understand *what* we are doing.

First, we need to be sure that an algorithm predicts well. If we do not understand the algorithm, how can we know this?

We can look at the predictions made against our validation data set. (I am sure some business stakeholders relying on models performing well would be horrified to learn they might have been tested only against a couple of hundred records to make sure they behaved as expected.) But this will not tell us if we have made [a catastrophic error in the selection of our data](https://www.reuters.com/article/us-amazon-com-jobs-automation-insight/amazon-scraps-secret-ai-recruiting-tool-that-showed-bias-against-women-idUSKCN1MK08G). It will not tell us if there are subsegments of our customer base for which the predictions do not work well. When the algorithm is a ‘black box’ which simply takes inputs and returns outputs with no clue as to its inner workings, the only debugging possible is to run many cases through and make sure the prediction is close to the truth.

Second, we need to be able to explain _why_ we made individual predictions. I am sure you would agree that it is not satistying to explain a refusal to accept a loan application on the grounds of ‘computer says no’. Neither the customer nor the customer service agent would have a reason to trust the algorithm. They are left to guessing why the application was not accepted.

![Computer says no](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.LHL6AyEugKKtdom-rCQTcQHaEV%26pid%3DApi&f=1)

But that is what we are asking people to do when we say ’trust the machine’. We are asking humans in a real process making operational decisions to use the predictions from a machine learning algorithm. Surely it would be in the interest of everyone invested in the model’s success to ensure the users trust and understand why the predictions are model. They need to get reasons of why this customer has been selected to receive this discount, or why a particular machine has been selected for maintenance. If they cannot trust it, adoption rates and compliance rates will remain low.

Third, there is an increasing focus on models being well governed and responsible. Initiatives like [GDPR](https://www.oreilly.com/radar/how-will-the-gdpr-impact-machine-learning/) prescribe how machine learning is governed, how users’ consent is managed and how interpretable a model’s predictions are. In parallel, there is increasing focus on [ethical AI](https://consult.industry.gov.au/strategic-policy/artificial-intelligence-ethics-framework/supporting_documents/ArtificialIntelligenceethicsframeworkdiscussionpaper.pdf) and [responsible AI](https://www.microsoft.com/en-us/ai/responsible-ai?activetab=pivot1:primaryr6), which include a set of principles to ensure machine learning models used by organisations are

* fair;
* reliable and safe;
* inclusive;
* transparent;
* private and secure; and
* have clear accountability.

### What have we lost?

With the classical statistical techniques of yore, we had measure of uncretainty around the parameters of the algorithm. We could tell where it was accurate and where it was less so. We could calculate the predictions with a modest amount of matrix mathematics.

The new techniques kind of forgot about these things. Or they added them as an afterthought.

Now, though, there is something of a renaissance brewing under the name of XAI—explainable AI. XAI typically refers to a suite of tools and technique that let us _interpret_ fitted models and _explain_ predictions of almost any model.

### SHAP

The new hero of the day is a set of libraries called [SHAP](https://shap.readthedocs.io/en/latest/)—**SH**apley **a**dditive **p**redictions.[^2]

[^2]: An in-depth treatment of this can be found in [Christoph Molnar’s e-Book _Interpretable Machine Learning_](https://christophm.github.io/interpretable-ml-book/).

What SHAP does, in essence, is to run a series of test observations through a model’s prediction algorithm to see what happens. The reason its use is increasing so rapidly is that it outputs a series of additive outputs. (You know, like the regression models do.)

And additive outputs are easy to interpret. You can add them up, take the mean (a statistician’s way of saying ‘average’); they behave sensibly and intuitively.

### Back on board the _Titanic_

So to show you how we can apply SHAP to the same data set as before, I used the Python [`XGBClassifier`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) against the data. I used the default parameters and the same feature transformations I used with the logistic regression model.

Instead of an analysis of deviance table, all the model fitting process gives me is a series of sequential trees that reduce the variation of error at each iteration. [This is what GBMs do.](https://en.wikipedia.org/wiki/Gradient_boosting) I can find out which features contribute to the model using one of the ‘afterthought’ techniques included with the algorithm.



![feature-importance](/assets/feature-importance.png)

**_The most important features as shown by the GBM afterthought._**

Note that in what is produced above, we get no sense of the bounds of error or variation.

To get some more information, we can turn to SHAP. It turns out that SHAP has a similar plot that also shows the individual points in the set of test observations we have used.

![shap-summary-plot](/assets/shap-summary-plot.png)

This is a bit more informative. We can see perfect separation on the basis of `Sex`; that there are some bad values of `Pclass`; that being old was not a good thing.

### Explaining predictions

As we did before, we can look at explaining individual predictions. SHAP can explain the prediction for the passenger most likely to survive.

![best-passenger](/assets/best-passenger.png)

Now that chart is easy to understand (for a data scientist).

Similarly for the unluckiest passenger.

![worst-passenger](/assets/worst-passenger.png)

> **Note**: with a bit more mathematics, we could have represented the outputs of the logistic regression in exactly the same way.

## Comparing the old and the new

So now you have seen what we used to do and what we can do now with SHAP.

“So,” I hear you saying, “you are building a second, additive and interpretable model to explain a model that used a technique that is inherently difficult to explain.”

Yes, that is right—now we have two models. One to predict, and one to explain the predictions and interpret the model.

“Wow,” you must be thinking, “these new-fangled machine learning techniques must be _really something_ for you to add the complexity of having two models instead of one.”

## Back to you, Kaggle

To show you how far we have come, I tested these models on Kaggle (and there are many, many models on Kaggle that predict better on this data set). Kaggle provides a test set of data that contains all the variables of the training data set, but does not include the information on whether an individual survived or not. Kaggle uses this unseen information to calculate a performance score. It uses a measure of accuracy, and higher means better.

First I ran the `XGBClassifier` model’s predictions through Kaggle. The result: 0.734 accuracy.

Next I tried the logistic regression. The result: 0.754 accuracy.

Looks like we need to do more work to explain our new, fancy machine learning techniques but get little gain in this instance.

The takeaway might be to think about your model development lifecycle:

  1. build an interpretable model using classical, interpretable techniques;
  2.  Once you are happy with this, use a modern machine learning technique to see if you get a significant gain in performance; and
  3. If you do, consider whether you want to spend more time tweaking the interpretable model or explaining the machine learning model. Most times you won't.

---

## Conclusion

Increasingly data scientists and the systems that use the models produced by data scientists are under pressure to be well-governed, interpretable and well-understood. Their predictions need to be, well, predictable.

The SHAP library is very useful and gives us insight into models that would otherwise have been obscured from our view. But for some use cases, you are better off using the classical techniques and understanding your model as you develop it. The alternative road can be a lot of complexity and effort for less understanding and no gain in prediction accuracy.

---
