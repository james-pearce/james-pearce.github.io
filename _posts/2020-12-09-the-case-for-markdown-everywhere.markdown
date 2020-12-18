---
layout: post
title:  "The case for Markdown everywhere"
date:   2020-12-09 09:00:00 +1100
categories: markdown
---


# _Why I love Markdown and want to see its use become more widespread_

## Intro

I _love_ writing in [Markdown](https://en.wikipedia.org/wiki/Markdown). It is simple to write, easy to remember and is widely adopted. When writing in Markdown, all I need to concentrate on is the writing and the keyboard. Markdown files are readable as text: everybody understands what you mean when you surround a word by asterisks — like \*\*really\*\*.

In this article, I go into why I like Markdown and why I would like to see support for it in all applications.

$$  $$

## What it is and a brief history

Does anyone remember old word processing applications (or programs) from the 1980s that ran on [MS-DOS](https://en.wikipedia.org/wiki/MS-DOS) like [WordStar](https://en.wikipedia.org/wiki/WordStar) and [WordPerfect](https://www.wordperfect.com/en/)? Before the dominance of [Microsoft Word][msword] and [WYSIWYG](https://en.wikipedia.org/wiki/WYSIWYG) on [Microsoft Windows](https://en.wikipedia.org/wiki/Microsoft_Windows) became the _de facto_ standard for document writing?

You wrote the document using special characters for formatting. If you wanted to preview the final product, you had to render the document to print or a special preview screen. But as long as you knew the characters to type, it was remarkably quick and efficient to compose a document. You were not stuck in the constant hell of switching between keyboard and mouse to compose and format a document.

![Some of the special characters appearing in a WordStar document on MS-DOS](https://upload.wikimedia.org/wikipedia/en/e/e3/Wordstar_Screenshot.png)

_Some of the special characters appearing in a WordStar document on MS-DOS._

<img src="https://upload.wikimedia.org/wikipedia/commons/0/00/Compaq_Portable_and_Wordperfect.JPG" alt="WordPerfect was perfect for word processing when computers looked like Archer technology" style="zoom:80%;" />

_WordPerfect was perfect for word processing when computers looked like [Archer][archer] technology._

Just while I am digressing … I recall from my childhood that word processors used to be actual computers themselves — computers that ran only one application. Offices had banks of typists who had upgraded from electric typewriters to word processors. I think the old equipment looks great now in a retro chic way.

![A Wang word processor really does have that Archer vibe](https://i.pinimg.com/originals/28/74/50/2874505128a2430de587030671d6d413.jpg)

_A Wang word processor really does have that Archer vibe._

<img src="https://cdn.newsday.com/polopoly_fs/1.5178733.1461864381!/httpImage/image.jpg_gen/derivatives/display_960/image.jpg" alt="The computers from Archer feature state of the art mechanical keyboards" style="zoom:80%;" />

_The computers from Archer feature state of the art mechanical keyboards._

According to the [Internet](https://en.wikipedia.org/wiki/Markdown), Markdown was created in 2004 by [John Gruber](https://en.wikipedia.org/wiki/John_Gruber) and the late, troubled [Aaron Swartz](https://en.wikipedia.org/wiki/Aaron_Swartz). The aim: a text format that could convert easily to valid [HTML][html] or [XHTML](https://en.wikipedia.org/wiki/XHTML).

The name ‘Markdown’ is a simple play on the term ‘markup’. [‘Markup’](https://en.wikipedia.org/wiki/Markup_language) refers to a language that describes a document’s formatting separately from its content. This makes it straightforward to change styles. Examples of popular markup languages are [HTML][html] and [LaTeX][latex] ($\LaTeX$).

Both of these languages have not achieved their full potential as markup languages, mainly because they have evolved to find widespread use in their specific niches beyond a simple requirement for document processing.

## Where can I use it and what is it good for?

Markdown has risen to prominence largely through its use on [GitHub](github.com), where is used for README files. [Github has its own flavour][gfm] and has [extensive guides](https://guides.github.com/features/mastering-markdown/) to help learn how best to use it. But learning Markdown is not that hard!

The basics of using Markdown are straightforward.

  * Use `#` for heading level 1, `##` heading level 2 and so one up to level 6 (please do not use this — I cannot keep your overcomplicated document structure in my head)
  * Surround text to be displayed in _italics_ by a single underscore (`_italics_`)
  * Similarly, use two asterisks (`**bold**`) for **bold**
  * Bullet points are created using two spaces and an asterisk

```markdown
  * bullet point 1
  * bullet point 2
    - subpoint 2.1
```

  * Numbered lists use a number followed by a full stop — Markdown will ignore the actual number you type, however

```markdown
1. Point 1
3. Point 2
3. Point 3
```

  * Unformatted text is specified using a backquote \`; you can format blocks of text using matching sets of triple backquotes (\`\`\`)
  * Block quotes of text are specified using a greater-than sign preceding the text

```markdown
> This is how you do a block quote

```

---

> This is how you do a block quote

---

### Analytics and maths

The popular analytical tools [Jupyter](https://jupyter.org) and [RStudio](https://rstudio.com) both use Markdown for writing reports that can include code, calculations, charts and other enhancements.

The other killer feature for someone like me is many versions support mathematics in the form of [LaTeX][latex] commands.
So it is easy to include a maths equation like $ y = X\beta + \epsilon $.

## Why I like it

### Structured documents

Using Markdown, or even markup languages, it is easy to create structured documents and _much harder_ to hard-code document-specific formatting than in [Word][msword]. This can be both a good and bad thing; I like it because it forces a writer to focus on the content rather than how the document looks.

### Workflow

I also find it much more intuitive and less disruptive to my workflow to include any formatting as characters while I type. Unlike [MS Word][msword], I am not jumping around between the keyboard and the mouse for formatting. This becomes even more important when you are working from a laptop and navigating between its keyboard and trackpad.

For those series writers, some software tools allow have a **focus mode** to help you concentrate. [Many old-school writers prefer writing using text editors only.](https://lifehacker.com/i-still-use-plain-text-for-everything-and-i-love-it-1758380840)

### Separation of text and formatting

Markdown describes the context and the structure of the formatting only, unlike with word processors like [MS Word][msword], where the formatting is enmeshed in the document itself. The separation makes it easy to change styles. [This is in line with good design practices.](https://en.wikipedia.org/wiki/Separation_of_content_and_presentation)

<img src="/assets/Text-and-style-together.svg" alt="Text and style confounded" style="zoom:80%;" />

_[MS Word][msword] combines text and style in the same document._

![Text and style separate](/assets/Text-and-style-separate.svg)

_Good design dictates that content formatting and style should be separate._

### Markdown files are plain text

Markdown files are text. This makes them **extremely portable** and editable with any text editor. But of course there are specialist tools that give you a bit more. Two of my favourites are [**Typora**][typora] and  [StackEdit](https://stackedit.io). Typora is available for Linux, Mac OS and Windows. StackEdit runs in the browser and integrates with your online storage account.

### Support for HTML

Markdown replaces the HTML characters `<` `>` and `&` with character entity references. This means you can do neat things like this: this text is <span style='color:blue'>blue</span> (`<span style='color:blue'>blue</span>`) or &frac12; (`&frac12;`) . HTML support is very useful for training documentation and writing technical manuals, for example, to type things like <kbd>Enter</kbd> (`<kbd>Enter</kbd>`). Because it resolves to HTML, it is also great for designing simple websites using tools like [Jekyll](https://jekyllrb.com). Using HTML makes Markdown _a little_ less usable, however.

### Writing code blocks

Markdown is also great for writing code; it supports a whole bunch of languages. Here is an example of how it renders  Python code.

```python
def add_one(x):
  y = x + 1
  # return the result
  return y
```

_Example of a basic Python function rendered in Markdown._

### Presentation slides

There are tools and frameworks that let you write Markdown to generate a series of slides and presentations. These can be pretty good and highly recommended when you need to include code in your presentation. The downside is that where you want diagrams and images, it seems less than natural to include them in Markdown — you pretty much end up with a frequent write–display–review workflow. Note though, that this mode of work is _very_ different from composing presentations in [PowerPoint](https://www.microsoft.com/en-us/microsoft-365/powerpoint) or [Prezi](https://prezi.com).

My favourite way of doing this is to use [RStudio][rstudio] to create [revealjs](https://bookdown.org/yihui/rmarkdown/revealjs.html) presentations.

## What could make it better

So you get that I love Markdown. But where could it be better? Improved support for tables, styles and images could improve Markdown, but I do not have great suggestions on how to design the improvements I would like to see.

### Tables

Writing tables in Markdown is okay. You basically write a text-based table like this.

```
| fruit  | colour | rating |
|--------|--------|-------:|
| banana | yellow | 10     |
| orange | orange |  5     |
| lemon  | yellow |  2     |
```

You do not even need to align all the heading and columns if you are lazy.

It will display as a nicely formatted table like this:

| fruit  | colour | rating |
| ------ | ------ | -----: |
| banana | yellow |     10 |
| orange | orange |      5 |
| lemon  | yellow |      2 |

I do like this — a lot. But sometimes you need detailed tables with multiple rows in a cell or merged cells. Markdown cannot cope with this natively. The workaround is to write the table in HTML, but it is painful (unless you use a helper tool) and the raw text is no longer readable.

### CSS

CSS can be used to provide custom styling and even some actions that control HTML. Without a good CSS tool, much of this is alien to me (being a simple user), so I am stuck with what we get from the tools as they convert the Markdown syntax to a display format. But some of you will be more capable than I am of combining the CSS with the Markdown converter to create nice looking outputs.

### Images

Image handling can be fiddly. If you want to include images as they are, it is straightforward — `![image text](image location)`. But to caption or zoom or anything else that might be nice, you need to go into HTML and use `<img>` tags. Again this messes with the readability of the raw text.

The other thing I do not like about images is how they are handled in the rendered documents. In some implementations of Markdown to HTML (or other destinations), the hard-coded links to images are maintained. This does not make a lot of sense for local files. You essentially need to load these to a destination that is accessible wherever the document is viewed.

## What would make it worse

What makes Markdown great for writing is its simplicity, its portability and its readability. If any extensions to the language change this, it will be for the worse.

### Portability

One of the enhancements that could reduce its portability is fragmentation into too many variants each with their own syntax. There are already several versions of Markdown in the world: two are [GitHub-Flavored Markdown](https://github.github.com/gfm/) and [PHP Markdown Extra](https://michelf.ca/projects/php-markdown/extra/) (which adds additional features not available in standard Markdown). If there are two many variants of Markdown its portability is reduced. You will not have a reliable way of knowing if a particular Markdown file will display correctly in another variant.

### Simplicity and readability

What makes Markdown simple compared to work is its limitations. The temptation will be to keep adding things to it: additional style elements, diagrams, improvements and so on. Additional style elements like CSS encroaching into the Markdown syntax reduces it simplicity _and_ the separate of content and style.

Adding diagrams (such as [Mermaid](https://mermaid-js.github.io/mermaid/#/) included in [Typora](typora)) seems useful, but has the downside of reducing the readability of the Markdown text except for the simplest of diagrams.



```mermaid!
graph LR
  A((A)) --> B((B))
  A --> C((C))
```

_Example of a Mermaid flowchart._

```mermaid
graph LR
  A((A)) --> B((B))
  A --> C((C))
```  
_The text version of the Mermaid flowchart is much less readable._

## What I want for Markdown

I love Markdown, and my world would be better if Markdown were supported **everywhere**!

By that I mean _all_ writing applications support Markdown syntax and extensions. Typing Markdown syntax would convert to the document’s format. Copy-and-paste commands would copy to Markdown, and intelligently convert pasted Markdown to the displayed formate.

I want to be able to type `**really**` in every application I use and see **really**. That includes [Word](msword) — which is still the standard despite open-source and free document alternatives (_kudos!_) and others like [Outlook](outlook), [Teams](https://teams.microsoft.com/), [OneNote]([https://www.onenote.com](https://www.onenote.com/)), wikis, [Wix]([https://www.wix.com](https://www.wix.com/)), [Medium]([https://medium.com](https://medium.com/)), [Atlassian’s Confluence](https://www.atlassian.com/software/confluence), [Gmail](https://gsuite.google.com/products/gmail/), [Apple mail](https://en.wikipedia.org/wiki/Apple_Mail), [Evernote](https://evernote.com/), [Google docs](https://docs.google.com/) and others. (To be fair, it kind of works in some of these, and in others you can apply extensions that add some of the functionality.)

## Conclusion

I really enjoy writing in Markdown because of the focus on writing content that is gives me. There are a couple of improvements that could be made, but given the opportunity of enhancing Markdown at the expense of its readability and simplicity, I would gladly keep it as is.

[html]: https://en.wikipedia.org/wiki/HTML
[msword]: https://en.wikipedia.org/wiki/Microsoft_Word
[outlook]: https://en.wikipedia.org/wiki/Outlook.com
[latex]: https://en.wikipedia.org/wiki/LaTeX
[rstudio]: https://rstudio.com
[typora]:  https://typora.io
[archer]: https://en.wikipedia.org/wiki/Archer_%282009_TV_series%29
[gfm]: https://github.github.com/gfm/

_I wrote the first draft of this article in Markdown. You can find a copy of the file [here](https://drive.google.com/file/d/16bg_tLdEJaTRdNclmfB23usqdyyHxfE0/view?usp=sharing)._

---

### More information

[Click here to access my **free course** on sentiment analysis.](https://3-crowns-academy.teachable.com/p/build-your-first-document-classifier-with-machine-learning-in-python)

&copy; 2020 James Pearce, 3 Crowns Consulting.
