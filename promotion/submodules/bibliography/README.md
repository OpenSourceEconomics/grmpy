# Bibliography

This repository hosts the central bibliography for numerous research projects.

The bibliography file is called ``literature.bib``. 

- ``annotated_bibliography`` We maintain an annotated bibliography of useful resources for future reference.
- For proper APA citation, only capitalise the first word of the title/heading and of any subtitle/subheading as well as proper nouns and certain other types of words. Use lowercase for everything else. (See also http://blog.apastyle.org/apastyle/2012/03/title-case-and-sentence-case-capitalization-in-apa-style.html and http://blog.apastyle.org/apastyle/2012/02/do-i-capitalize-this-word.html)
- The necessary commands for proper APA citation in LaTeX seem to differ between operating systems. **For Windows and Ubuntu users**, **header.tex** has to include the following commands:
  - *\usepackage{apacite}* 

- **For Mac users**, the following commands have to be included:
  - *\RequirePackage{bibentry}*
  - *\makeatletter\let\saved@bibitem\@bibitem\makeatother*

  as well as:

  - *\usepackage [apaciteclassic]{apacite}* 
  - *\usepackage{notoccite}* 
  - *\usepackage{bibentry}* 

- The **main.tex** has to include the following commands (regardless of operating system):

  - *\bibliographystyle{apacite}*
  - *\bibliography {../../../ submodules/bibliography/literature}*

- If you want to insert a new entry in ``literature.bib`` via JabRef, please do not insert the BibtexKey manually but instead follow the following steps:

  - Go to **Options** -> **Preferences** -> **BibTeX key generator**
  - Insert **[auth].[year]** as the default pattern
  - Check the box **Generate keys before saving (for entries without a key)**

  This will create the keys automatically and in compliance with the other entries in ``literature.bib``.

