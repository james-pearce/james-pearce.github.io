---
layout: post
title:  "You should be using Azure Machine Learning designer"
date:   2020-11-10 09:00:00 +1100
categories: machine-learning azure
---

# _A modern update with a familiar look and feel gives data scientists the best of both worlds_

I have been a strong advocate of Microsoft’s [Azure Machine Learning Studio (classic)](https://studio.azureml.net/) since its release in 2015. It is a great tool, parts of it remain relevant and it is an excellent platform for students new to machine learning.

It has an outstanding and its intuitive visual interface make it easy to use. It can be extended to be as powerful as you need it to be with R and Python scripts. Its ties to R are strong — this reflects Microsoft’s investment in R at the time. Microsoft published a slew of [galleries and examples](https://gallery.azure.ai/) that made Azure ML Studio an exceptional machine learning resource. It was — and still is — a go-to place to see how someone else has solved a problem in a way that is easy to understand.

![A linear regression experiment to predict car prices in Azure Machine Learning Studio (classic)](/assets/azure ML classic 01.png)  
_A linear regression experiment to predict car prices in Azure Machine Learning Studio (classic)_

Fast forward to late 2020 and the world has moved on. Microsoft has launched [an impressive range of machine learning services under the Azure banner](https://azure.microsoft.com/en-us/services/machine-learning-service/), with strong ties to Python libraries. [**They have now made them essentially free.**](https://azure.microsoft.com/en-us/pricing/details/machine-learning/) The focus of machine learning has tilted slightly [towards Python and away from R](https://www.kdnuggets.com/2019/05/poll-top-data-science-machine-learning-platforms.html). Microsoft has started to deprecate its development in Azure Machine Learning Studio (classic) in favour of its new suite of machine learning services. The direction for R is through [integration with SQL Server](https://docs.microsoft.com/en-us/sql/machine-learning/sql-server-machine-learning-services) (named SQL Server Machine Learning Services).

Is it me, or are the names confusing?

| Azure  Machine Learning Studio  (classic) | A  visual machine learning environment                       |
| ----------------------------------------- | ------------------------------------------------------------ |
| Azure  Machine Learning Services          | Azure-based  API and integrated Python libraries for the full machine learning lifecycle |
| SQL Server  Machine Learning Services     | R  and Python machine learning in SQL Server                 |



Come on Microsoft, bring back names we can relate to. Like [Clippy](https://www.theverge.com/2019/3/22/18276923/microsoft-clippy-microsoft-teams-stickers-removal).

![Fans of Apple were envious of the Windows-only Clippy](http://cdn.windowsreport.com/wp-content/uploads/2014/10/clippy-windows-8-10.jpg)  
_Fans of Apple were envious of the Windows-only Clippy_

With all of this, faithful users were left with a choice: do I want ease of use or tight integration with Azure? You could have flexibility in Azure environments but were limited to code and the learning curve that comes with that. Or you could have easy development with a brilliant visual interface, but were limited in the Azure environments in terms of scale, compute, regions and storage for deploying models.

![The tradeoffs you had to make between Azure ML Studio and Azure ML Service.](/assets/how it was 2.png)

Enter the new visual interface for Azure Machine Learning known as (I think) [Azure Machine Learning designer](https://docs.microsoft.com/en-us/sql/machine-learning/sql-server-machine-learning-services) which is part of the new and equally-confusingly-named Azure Machine Learning Studio. If you are familiar with Azure Machine Learning Studio (classic), you will know how to use the new interface. It looks exactly the same _and_ it shares many features with its predecessor.

Look at the screen grab below of the new Azure Machine Learning designer interface: the interface we love is back — with upgrades!

![Azure Machine Learning designer](/assets/Azure ML designer.png)

Things are not quite the same. The ties with R have been replaced by stronger — and better — ties to Python. You have full control over your compute environment, and you have access to the full Azure run history of experiments. So you have flexibility with Azure, *plus* a great visual coding interface which you probably already know.

![Details are available for your visual experiments.](/assets/Azure run experiment.png)

## Conclusion

To sum up, the new visual interface for Azure Machine Learning Service is a great addition to Microsoft’s Azure machine learning suite. You get the best of both worlds: power, flexibility and an intuitive interface **plus** all the bells and whistles from the Azure Machine Learning suite. These include [automated machine learning](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml), [interpretable machine learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability) and [model monitoring](), all important capabilities for a modern machine learning suite.

![azure ml designer 2](/assets/azure ml designer 2.png)

_Now Azure has an option that gives you power and flexibility with a visual interface_

---

### Free sentiment analysis course

[Click here to access my **free course** on sentiment analysis.](https://3-crowns-academy.teachable.com/p/build-your-first-document-classifier-with-machine-learning-in-python)
