---
layout: post
title:  "Like sentiment analysis can detect sarcasm!"
date:   2020-11-08 09:00:00 +1100
categories: NLP sentiment
---


_Automatic sentiment analysis of texts has historically struggled to handle sarcasm well. Have advances in machine learning techniques made inroads to the detection of sarcasm?_

You may have heard about [sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis). It is a technique that analyses text and tries to establish whether the emotional state of the person who wrote it was **positive** or **negative**. It is a powerful way of understanding whether tweets regarding your product are trending positively or negatively. In turn, you can use this as an early warning system for problems as they arise. And there are myriad other uses.

But English can be a strange language. One of the toughest jobs that automated sentiment detection has is when sarcasm is involved.

This is not surprising. Actual people can have difficulty in detecting sarcasm. Think of Sheldon from _Big Bang Theory_ or Sean Murphy from _The Good Doctor_. (And some have argued that Alanis Morrisette cannot recognise irony, [but maybe those people are too pedantic](https://www.theatlantic.com/notes/2016/05/alanis-morissette-recognizes-its-not-ironic/481875/).)

<img src="https://znculturecast.files.wordpress.com/2013/03/dc056b4f8c9493f3c64e6e0a85382b31.jpeg" style="zoom:67%;" />  
_Dr Sheldon Cooper is really good at detecting sarcasm_

Consider the following quote.

> I really enjoyed this book and will recommend it to my friends and family.

This quote contains _positive_ words (‘enjoyed’ and ‘recommended’) and reflects positive emotion.

But consider this similar quote.

> I could not put this book down and will not hesitate to recommend it to family and friends.

This sentence is a bit different.

| positive | negative |
|----------|----------|
| recommend | not (twice) |
|           | down    |
|           | hesitate |

Although overall it is a positive review, it contains some negative words. Simple sentiment calculations based on the occurrence of positive or negative words will misclassify this review.

There are three negative words but only one positive word, giving a net negative result overall.

Now consider a sarcastic review.

> I am glad the author chose to use overly long and flowery descriptions for each and every change of scene as I needed some scrap paper to line my pet parrot's cage. Furthermore, the huge number of typographic errors was a refreshing change from reading professionally edited books. I recommend this to anyone who also enjoys long and unnecessary dental procedures.

This is full of positive words — 'glad', 'huge', 'refreshing', 'recommend' and 'enjoys' — and contains _no_ negative words. Taken as a whole the review uses humourous sarcasm to give the book a negative review.

This presents a huge challenge to standard sentiment calculation techniques.

Over the past ten or so years, improvements in machine learning algorithms have brought us so much. Computer algorithms can now  [translate at a rate comparable to human translators](https://www.forbes.com/sites/bernardmarr/2018/08/24/will-machine-learning-ai-make-human-translators-an-endangered-species/#45b3d2df3902), [do a better job than Hollywood](https://deepfakenow.com/hollywood-deepfake-movies/) of reanimating those actors who are no longer with us, [caption images](https://www.microsoft.com/en-us/research/blog/novel-object-captioning-surpasses-human-performance-on-benchmarks/) at a better success rate than people and generate [fascinating definitions](https://www.thisworddoesnotexist.com) for non-words.

Understanding whether comments are sarcastic can be useful to know: some have suggested that sarcastic users may have more influence than other users. The question I want to explore further is:

##### Have recent advances in machine learning allowed algorithms to correctly and reliably detect sarcasm?

Before getting too far into this topic, what _is_ sarcasm? You can look it up on [Duck Duck Go](https://duckduckgo.com) if you like, but here is my take on it.

Sarcasm is a form of irony used to convey the opposite of what is stated in some sense. It can be used as [the lowest form of wit](https://www.independent.co.uk/news/science/sarcasm-how-lowest-form-wit-actually-makes-people-brighter-and-more-creative-10416281.html) or to be humorous; it is often used as an insult or to mock but also to display irritation. In conversation, sarcasm is usually accompanied by gestures, tonal stress or eye-rolls.

When analysing text in situations where we do not get the benefits of tonal or visual cues, we need to rely _only_ on what was written. Sometimes a writer will include additional clues like emoticons and emoji to signal their true emotion, which can help.

One of the problems with detecting sarcasm is having some baseline of truth against which to compare. We need some data on what comments are sarcastic and which are not. As sarcasm is not universally understood — and we know Sheldon would not be an effective arbiter of sarcasm — any pre-existing data set with sarcastic–non-sarcastic labels is bound to contain errors.

A relatively recent data set by Oprea and Magdy [(_Reactive Supervision_)](https://arxiv.org/pdf/2009.13080.pdf) contains self-described sarcastic comments along with an explanation of why the author thought the comment was sarcastic. I will be a useful resource for training and evaluating sarcasm detectors, but even this will present some problems:  the data set does not include the context of the original comment, and [people themselves are unreliable at reporting sarcasm in their own comments](http://documents.mx/documents/yeah-right-a-linguistic-analysis-of-self-reported-sarcastic-messages-and-their-contexts.html).

There is a lesson here for aspiring writers. Avoid sarcasm if you want to be understood well: your audience may not comprehend the signal you are hiding in the noise of sarcasm.

Current strategies to detect sarcasm in comments generally look to explore knowledge outside of the comment itself (I use the word ‘comments’ here to include reviews, tweets and other short texts). For example,

* [separate the analysis of emotion from the sentiment analysis](www.crowdanalyzer.com);
* [understand the context in which the comment was made](https://deepai.org/publication/the-role-of-conversation-context-for-sarcasm-detection-in-online-interactions);
* [examine whether an author has been sarcastic in the past](https://towardsdatascience.com/sarcasm-detection-with-nlp-cbff1723f69a);
* determine whether the comment seems to contradict itself and contains many positive and negative words;
* determine whether the comment seems to contradict generally known facts (sometimes split into [numeric](https://www.toptal.com/deep-learning/4-sentiment-analysis-accuracy-traps) and non-numeric comments); and
* compare the structure of the comment with known structures of sarcasm (for example, [going from positive to negative]([here](https://www.crowdanalyzer.com/blog/sentiment-analysis)), [a polite but contrasting tone](https://towardsdatascience.com/sarcasm-detection-with-nlp-cbff1723f69a) and [excessive use of capitals and exclamation marks](https://www.themarysue.com/sarcasm-detecting-algorithm-online/)).

And of course, you could try multiple approaches. (See [here](https://www.researchgate.net/publication/325843750_A_COMPREHENSIVE_STUDY_ON_SARCASM_DETECTION_TECHNIQUES_IN_SENTIMENT_ANALYSIS/link/5bb0db42a6fdccd3cb7f80e0/download) for some more detail on sarcasm structure.)

A complementary approach is to use [deep learning architectures](https://github.com/SenticNet/CASCADE--ContextuAl-SarCAsm-DEtector) to train sarcasm detectors. These are showing some promise, especially if combined with some of the approaches above. But the results in any of the papers are not blowing me away in the same way that other deep learning models do.

It would seem from the literature, code, blog posts and other informative sources that we have made some inroads into sarcasm detection, but there is still a way to go.

For some fun, I thought I would try an extremely non-scientific approach to evaluating two online sarcasm detectors. In both of these online detectors, you can visit the website and enter your text. (Obviously these detectors cannot account for context.) To test them out, I used two sarcastic book reviews from Amazon as well as the example from above.

Here are the results:

| Comment                                                      | [The sarcasm detector](http://www.thesarcasmdetector.com) (the score is from -100 to 100; the higher the more sarcastic) | [Parallel Dots API](https://www.paralleldots.com/sarcasm-detection)  (0% to 100%; the higher the more sarcastic) |
| ------------------------------------------------------------ | -----------------------------------------------------------: | -----------------------------------------------------------: |
| I am glad the author chose to use overly long and flowery descriptions for each and every change of scene as I needed some scrap paper to line my pet parrot's cage. Furthermore, the huge number of typographic errors was a refreshing change from reading professionally edited books. I recommend this to anyone who also enjoys long and unnecessary dental procedures. |                                                          -46 |                                                          44% |
| This book works if you need something to read on a plane trip from Florida to Ohio.  It held my interest to an extent.  However, I got the feeling he knocked this one off in about the time it takes to type it.  It takes about 3 minutes to figure out his daughter is coming home for Christmas.  Wait for the paperback. |                                                          -71 |                                                          58% |
| One need only to have listened to the oblique babbling of most corporate managers to realize that this is their Bible. Admittedly it will help your career.  You will learn how to speak out of both sides of your mouth, appear agreeable at all times, and engage in all manner of corporate BS.  Everyone will like you, except for those ne\'er-do goods-who abhor pretension and deceit.  And, most importantly, you will get that raise!  After all, get real, being honest, principled and lucid won\'t pay the rent and may even get you a pink slip. If you want to "get ahead" buy this book! If you are like me and amuse yourself by reading the kind of obfuscate and dissimulating language found in those emails from managers that arrive in your workplace computer, get this book for a good laugh! Dale Carnegie is the St. Paul of American Yuppies. |                                                          -36 |                                                          27% |

The commercial version from Parallel Dots worked a little better than _The Sarcasm Detector_, but both are obviously struggling with the meaning of these reviews.

## Conclusion

Sarcasm detection has improved but can get better.

There is potential in combining various approaches, using deep learning architectures, accounting for the conversation and user context and codifying patterns of known sarcasms to improve the current state of detection.

But keep in mind the point is moot. Sarcasm is only an effective means of communication when the auther and reader share context. Without this, humans — not just algorithms — will fail to get the intended message.

If you are analysing sentiment, make sure you use the emotion of any included emoji or emoticons to help your analysis. Include a rating field if you want better understanding of your customers’ true sentiment.

If you are a writer, _be clear_.

---

### More information

[Click here for more information on sentiment analysis and to access my **free course**.](https://3-crowns-academy.teachable.com/p/build-your-first-document-classifier-with-machine-learning-in-python)
